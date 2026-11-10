# Перенос XFeat на NPU RK3588: экспорт, конвертация, скорость

Модель: исходный XFeat (`third_party/accelerated_features/weights/xfeat.pt`), вход 640×480 (и 480×640 для портретных кадров).

## Итог шагов 1–4

- Основная часть XFeat (свёрточная сеть и головы) экспортирована в ONNX и сконвертирована в RKNN для RK3588 в двух вариантах: FP16 и INT8.
- Выходы ONNX совпадают с PyTorch до 3·10⁻⁵. FP16 в симуляторе совпадает с ONNX (cos = 1.00000). INT8: cos ≈ 0.995, есть заметные локальные ошибки.
- Скорость сети на NPU (3 ядра): **FP16 — 12.05 мс (83 FPS), INT8 — 8.64 мс (116 FPS)**.
- INT8 быстрее FP16 только в 1.4 раза. Вероятно, сеть упирается в пропускную способность памяти, а не в вычисления.
- Качество на HPatches (симулятор): **FP16 не теряет качество** относительно float32. **INT8 теряет 6% совпадений, 1.7 пункта MMA@3 и 0.8 пункта AUC@3**.
- **Решение: основной вариант — FP16.** INT8 пока остаётся дополнительным, его можно попробовать улучшить (`mmse`, гибридное квантование).

## Разделение модели между NPU и CPU

| Часть | Где выполняется | Почему |
|---|---|---|
| `mean` по каналам и `InstanceNorm2d(1)` | CPU, `deploy/preprocess.py` | статистику нужно сначала посчитать по всему кадру; на CPU это одна строка numpy |
| `block1`…`block5`, `block_fusion`, `heatmap_head`, `keypoint_head` | **NPU** | свёртки, BN, ReLU, bilinear-интерполяция |
| softmax, NMS, top-k, выборка дескрипторов, L2-нормализация, MNN-сопоставление | CPU | число точек меняется от кадра к кадру |
| `fine_matcher` (refinement) | не переносился | по заданию не трогаем, ≈1% стоимости |

Параметры: часть на NPU — 657 910 (у `config_A` — 395 766); `fine_matcher` — 886 848.

Изменения в графе. Оба численно эквивалентны оригиналу:
1. Нормализация входа вынесена из графа: на вход ONNX/RKNN подаётся уже нормализованное изображение `(1, 1, H, W)`.
2. `_unfold2d(x, 8)` (`unfold` + 5-мерный `permute`) заменён свёрткой 8×8, шаг 8, с 64 one-hot фильтрами (равна `pixel_unshuffle(x, 8)`, проверено `torch.equal`). Свёртка удобна для NPU, а её веса 0/1 квантуются в INT8 без потерь.
3. Размер для `F.interpolate` задан константой (`H/8 × W/8`), чтобы в графе не было `Shape/Slice/Concat`.

## Шаг 1. Экспорт в ONNX

Скрипт `deploy/export_onnx.py`, окружение conda `xfeat` (torch 2.14.1), классический экспортер (`dynamo=False`), opset 13, `ir_version = 8`.

| Проверка | feats | keypoints | heatmap |
|---|---|---|---|
| оригинал ↔ обёртка (torch), max\|diff\| | 5.96e-06 | 2.34e-05 | 8.64e-07 |
| torch ↔ onnxruntime, max\|diff\| | 6.68e-06 | 3.29e-05 | 1.34e-06 |

Выходы: `feats` 1×64×60×80, `keypoints` 1×65×60×80, `heatmap` 1×1×60×80. Операции в графе: `Conv, Relu, Add, AveragePool, Resize, Sigmoid` (BN встроен в свёртки). `config_A` экспортируется так же, расхождения того же порядка.

## Шаг 2. Конвертация в RKNN

Скрипт `deploy/convert_rknn.py`, conda-окружение `rknn` (rknn-toolkit2 2.5.0), `target_platform=rk3588`, `optimization_level=3`, без `mean/std` (вход уже нормализован).
Калибровка INT8: `deploy/make_calib.py`, 50 нормализованных кадров `1.ppm` из последовательностей HPatches, взятых равномерно по списку.

Сравнение симулятора с onnxruntime на `v_tabletop/1.ppm`:

| | FP16 cos | FP16 max\|diff\| | INT8 cos | INT8 max\|diff\| | Диапазон |
|---|---|---|---|---|---|
| feats | 1.00000 | 0.012 | 0.99549 | 1.28 | ±7.5 |
| keypoints | 1.00000 | 0.105 | 0.99473 | 4.78 | −35…10 |
| heatmap | 1.00000 | 0.0016 | 0.99614 | 0.135 | 0.01…0.73 |
| дескрипторы, косинус по ячейкам | 1.00000 | | 0.99549 | | |

Размер файлов: FP16 1.98 МБ, INT8 1.26 МБ.

## Шаг 3. Скорость на плате

`rknn_benchmark` (замеряется только `rknn_run`, 100 прогонов после 5 прогревочных), governor NPU `rknpu_ondemand` (800 МГц под нагрузкой).

| Модель | 1 ядро | 2 ядра | 3 ядра | Веса | Внутренние буферы |
|---|---|---|---|---|---|
| FP16 | 22.42 мс / 44.6 FPS | 15.28 мс / 65.4 FPS | 12.05 мс / 83.0 FPS | 1.41 МБ | 21.8 МБ |
| INT8 | 15.96 мс / 62.6 FPS | 10.95 мс / 91.3 FPS | 8.64 мс / 115.8 FPS | 0.75 МБ | 20.6 МБ |

Наблюдения:
- INT8 / FP16 ≈ 1.4× при любом числе ядер. Внутренние буферы при INT8 почти не уменьшились (20.6 против 21.8 МБ): первые слои работают на картах 480×640 и 240×320, поэтому, вероятно, узкое место — память. Подтвердить это может только послойный профиль.
- Три ядра ускоряют в ≈1.86× (у MobileNetV2 было 1.66×).
- Для сравнения: на ПК `extract` (вместе с постобработкой) занимает 35 мс на CPU (4 потока) и 4.1 мс на RTX 3050 (`results/speed_*_640x480_*.json`). Напрямую с цифрами NPU это не сравнить: на NPU измерена только сеть.
- Замер сделан на `.rknn`, собранных из ONNX до фиксации размера в `interpolate` (в графе ещё были `Shape/Slice/Concat`). Скорость нужно перемерить на моделях из нового ONNX.

## Шаг 4. Качество на HPatches

Скрипт `deploy/hpatches_rknn.py`. Протокол тот же, что у `benchmark/hpatches_eval.py`: 580 пар, короткая сторона 480, MNN (`min_cossim=-1`), top-k 4096, MAGSAC с порогом 3 px, 10 000 итераций и 5 сидов. Метрики: MMA@1…10, AUC@3/5/10 ошибки углов, MHA@3/5/7.

Отличия от исходной оценки:
- Сеть работает на **фиксированном входе**. Альбомные кадры подаются в модель 640×480, портретные в 480×640. Кадр растягивается до размера модели, а координаты точек пересчитываются обратно (как `rh/rw` в `XFeat.preprocess_tensor`). В HPatches 222 из 696 кадров имеют размер 480×640, 97 кадров портретные.
- Выходы сети берутся из onnxruntime или из симулятора RKNN. Постобработка (softmax, NMS, top-k, выборка дескрипторов) — исходный код `XFeat.detectAndCompute`, на CPU.
- Признаки сначала извлекаются для всех кадров и кэшируются (`deploy/cache/`), потом сопоставляются. Поэтому одновременно существует только один экземпляр RKNN.

Проверка скрипта: на двух последовательностях ONNX и исходный torch дают почти одинаковые результаты по парам (например, `i_ajuntament` 1–2: 2731 и 2727 совпадений, MMA@3 0.807 и 0.807).

### Результаты (все 580 пар)

| Вариант | Совпадений | MMA@3 | AUC@3 | AUC@5 | AUC@10 | MHA@3 i | MHA@3 v |
|---|---|---|---|---|---|---|---|
| torch, родной размер (подзадача 2) | 2004 | 64.5 | 58.3 | 71.0 | 82.6 | 96.1 | 77.6 |
| onnx fp32, фиксированный размер | 2002 | 64.2 | 57.9 | 70.6 | 82.4 | 95.4 | 77.1 |
| rknn fp16 (симулятор) | 2001 | 64.2 | 58.1 | 70.6 | 82.4 | 95.5 | 77.2 |
| rknn int8 (симулятор) | 1876 | 62.5 | 57.3 | 70.1 | 82.0 | 95.1 | 75.7 |

### Цена каждого шага

| Переход | Совпадения | MMA@3 | AUC@3 | AUC@10 | MHA@3 i | MHA@3 v |
|---|---|---|---|---|---|---|
| родной размер → фиксированный | −0.1% | −0.3 | −0.4 | −0.2 | −0.7 | −0.5 |
| fp32 → fp16 (NPU) | 0% | 0.0 | +0.2 | 0.0 | +0.1 | +0.1 |
| fp16 → int8 (NPU) | **−6.2%** | **−1.7** | −0.8 | −0.4 | −0.4 | **−1.5** |

Разброс AUC между сидами MAGSAC составляет примерно ±0.2–0.4 пункта, поэтому отличия меньше ~0.5 пункта считаем шумом.

### Выводы

1. **FP16 на NPU работает без потерь.** Все метрики совпадают с float32 в пределах шума.
2. **Фиксированный вход стоит 0.2–0.7 пункта.** Это потеря от растяжения части кадров до 640×480 / 480×640, NPU здесь ни при чём. В сумме FP16-модель на плате почти не отличается от исходного XFeat.
3. **INT8 сильнее всего меняет детектор.** Падают число совпадений и MMA@3. Это согласуется с локальными ошибками `keypoints`/`heatmap` из шага 2: часть точек не проходит порог 0.05 или смещается. Оценка гомографии страдает меньше (AUC@3 −0.8, AUC@10 −0.4), потому что MAGSAC отбрасывает лишние совпадения. Сильнее всего страдают пары со сменой ракурса (`v`), на смене освещения (`i`) потери в пределах шума.
4. По заранее заданному критерию (INT8 подходит, если теряет меньше ~1 пункта и по AUC@3, и по MMA@3) **INT8 не проходит**: AUC@3 в пределах нормы, MMA@3 нет.

| | Время сети, 3 ядра | AUC@3 относительно fp32 | MMA@3 относительно fp32 |
|---|---|---|---|
| FP16 | 12.05 мс (83 FPS) | +0.2 | 0.0 |
| INT8 | 8.64 мс (116 FPS) | −0.6 | −1.7 |

**Решение:** серию для прореженных конфигураций (`config_A`…`D`) выполнять в FP16. Тогда изменения метрик отражают только прореживание и не смешиваются с ошибкой квантования.

## Проблемы и решения

| Проблема | Решение |
|---|---|
| Прореженные модели сохранены целиком (`torch.save(net)`) и требуют torch 2.14, а в `rknn` стоит torch 2.4 | Экспорт в окружении `xfeat`, в `rknn` передаётся только `.onnx` |
| В графе были `Shape/Slice/Concat` из-за `x3.shape` в `interpolate` | Размер задан константой в конструкторе обёртки |
| Число `params` включало `fine_matcher`, которого нет в ONNX | Подсчёт без `fine_matcher` |
| `build: The input ... shape (1, 480, 640, 1) is wrong, expect 'nchw'` | Калибровочные `.npy` в rknn-toolkit2 нужны в форме входа ONNX — NCHW `(1, 1, H, W)` |
| `rknn_benchmark` на плате читает `.npy` в NHWC | Для бенчмарка отдельный файл `bench_input_nhwc.npy` (`transpose(0, 2, 3, 1)`) |
| Кадры HPatches бывают разного размера и ориентации, а модель RKNN имеет фиксированный вход | Две модели (640×480 и 480×640), растяжение кадра и обратный пересчёт координат |
| Симулятор запускает только модель, собранную в той же сессии (`load_onnx` + `build`) | `hpatches_rknn.py` сам собирает RKNN из ONNX с теми же настройками, что `convert_rknn.py` |
| `ModuleNotFoundError: No module named 'rknn'` | Прогоны `--backend rknn` запускаются в окружении `rknn`, а не `xfeat` |

## Команды

```bash
# окружение xfeat (экспорт)
conda create -n xfeat python=3.12 -y && conda activate xfeat
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install torch-pruning==1.6.1 numpy opencv-python matplotlib tqdm onnx onnxruntime

# шаг 1: экспорт (из корня репозитория)
python -m deploy.export_onnx
python -m deploy.export_onnx --model prune/results/model_pruned/config_A.pth --name config_A

# шаг 2: калибровка и конвертация
conda activate rknn
python -m deploy.make_calib
python -m deploy.convert_rknn --onnx deploy/models/xfeat_640x480.onnx --dtype fp16
python -m deploy.convert_rknn --onnx deploy/models/xfeat_640x480.onnx --dtype i8 --dataset deploy/calib/640x480/dataset.txt

# шаг 3: скорость
python -c "import numpy as np, glob; f=sorted(glob.glob('deploy/calib/640x480/*.npy'))[0]; np.save('deploy/models/bench_input_nhwc.npy', np.load(f).transpose(0,2,3,1).astype(np.float32))"
ssh rk3588 'mkdir -p /root/xfeat'
scp deploy/models/xfeat_640x480_fp16.rknn deploy/models/xfeat_640x480_i8.rknn deploy/models/bench_input_nhwc.npy rk3588:/root/xfeat/
# на плате:
cd /root/rknn_benchmark_Linux && export LD_LIBRARY_PATH=./lib
for m in fp16 i8; do for c in 1 3 7; do
  ./rknn_benchmark /root/xfeat/xfeat_640x480_$m.rknn /root/xfeat/bench_input_nhwc.npy 100 $c | grep -E "total|Avg"
done; done

# шаг 4: качество на HPatches
conda activate xfeat
python -m deploy.export_onnx                              # 640x480 (переэкспорт с постоянным размером)
python -m deploy.export_onnx --height 640 --width 480     # 480x640 для портретных кадров
python -m deploy.hpatches_rknn --backend onnx             # float32-эталон (можно и в rknn)
conda activate rknn
python -m deploy.make_calib --height 640 --width 480
python -m deploy.hpatches_rknn --backend rknn --dtype fp16
python -m deploy.hpatches_rknn --backend rknn --dtype i8
# результаты: results/hpatches_xfeat_{onnx,rknn_fp16,rknn_i8}_all_r480_thr3.json
```

## Файлы

| Файл | Назначение |
|---|---|
| `deploy/preprocess.py` | нормализация входа (общая для окружений `xfeat` и `rknn`) |
| `deploy/export_onnx.py` | PyTorch → ONNX + проверка выходов |
| `deploy/make_calib.py` | калибровочный набор INT8 |
| `deploy/convert_rknn.py` | ONNX → RKNN + проверка в симуляторе |
| `deploy/hpatches_rknn.py` | оценка на HPatches через onnxruntime или симулятор RKNN |
| `deploy/models/` | `.onnx`, `.rknn` (не в git) |
| `deploy/calib/`, `deploy/cache/` | калибровочные `.npy` и кэш признаков (не в git) |
| `results/hpatches_xfeat_*_all_r480_thr3.json` | результаты шага 4 |
| плата: `/root/xfeat/`, `/root/rknn_benchmark_Linux/` | модели и бенчмарк |

## Открытые вопросы

- Перемерить скорость на плате для `.rknn`, собранных из нового ONNX (с постоянным размером).
- Можно ли вернуть качество INT8: `quantized_algorithm="mmse"`, а если не поможет, гибридное квантование с головами `keypoint_head`/`heatmap_head` в FP16.
- Калибровка на кадрах HPatches пересекается с тестовым набором. Для итоговых цифр её стоит заменить кадрами не из HPatches (MegaDepth).
- Послойный профиль NPU нужен, чтобы подтвердить, что узкое место — память.
- Скорость конфигураций `config_A`…`config_D` на NPU.
