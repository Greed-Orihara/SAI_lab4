import csv
import math
import random
from pathlib import Path
from statistics import median
from time import perf_counter

POINTS = [
    (2, -15), (12, 1), (12, -18), (13, -17), (13, 3),
    (11, -2), (9, 1), (-11, 17), (5, -15), (-10, 18),
    (1, -17), (13, -19), (16, 0),
]
NEURONS = 4
EPOCHS = 80
MAX_EPOCHS = 300
STABILITY_LIMIT = 0.005
STABLE_EPOCHS = 5
SEED = 14
REPEATS = 21
LEARNING_START = 0.5
LEARNING_END = 0.02
FATIGUE_START = 0.4
RECOVERY = 0.8


def normalize(vector):
    x, y = vector
    length = math.hypot(x, y)
    return x / length, y / length


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def winner(point, weights, fatigue):
    best = 0
    best_score = -math.inf
    for index, weight in enumerate(weights):
        score = dot(point, weight) - fatigue[index]
        if score > best_score:
            best = index
            best_score = score
    return best


def train(points, initial_weights, orders, use_fatigue, check_convergence=False):
    weights = initial_weights.copy()
    fatigue = [0.0] * NEURONS
    wins = [0] * NEURONS
    stable_epochs = 0
    previous_groups = None
    no_penalties = [0.0] * NEURONS
    for epoch, order in enumerate(orders, 1):
        if check_convergence:
            previous_weights = weights.copy()
        progress = min(1.0, (epoch - 1) / (EPOCHS - 1))
        rate = LEARNING_START + (LEARNING_END - LEARNING_START) * progress
        penalty = FATIGUE_START * (1 - progress)
        for index in order:
            point = points[index]
            best = winner(point, weights, fatigue)
            wins[best] += 1
            wx, wy = weights[best]
            weights[best] = normalize((wx + rate * (point[0] - wx),
                                       wy + rate * (point[1] - wy)))
            if use_fatigue:
                for j in range(NEURONS):
                    fatigue[j] *= RECOVERY
                fatigue[best] += penalty
        if check_convergence:
            largest_move = max(math.dist(old, new) for old, new in zip(previous_weights, weights))
            groups = [winner(point, weights, no_penalties) for point in points]
            if largest_move <= STABILITY_LIMIT and groups == previous_groups:
                stable_epochs += 1
                if stable_epochs == STABLE_EPOCHS:
                    return weights, wins, epoch
            else:
                stable_epochs = 0
            previous_groups = groups
    return weights, wins, None


def benchmark(points, initial_weights, orders):
    results = {}
    times = {False: [], True: []}
    for repeat in range(REPEATS):
        # Чередуем порядок замеров, чтобы один режим не запускался первым всегда.
        modes = (False, True) if repeat % 2 == 0 else (True, False)
        for use_fatigue in modes:
            started = perf_counter()
            results[use_fatigue] = train(points, initial_weights, orders, use_fatigue)
            times[use_fatigue].append((perf_counter() - started) * 1000)
    return results, times


def summarize(points, weights):
    labels = []
    error = 0.0
    for point in points:
        best = winner(point, weights, [0.0] * NEURONS)
        labels.append(best)
        wx, wy = weights[best]
        error += math.hypot(point[0] - wx, point[1] - wy)

    centroids = {}
    scaled = {}
    for index, weight in enumerate(weights):
        cluster = [p for p, label in zip(POINTS, labels) if label == index]
        if not cluster:
            continue
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        centroids[index] = (cx, cy)
        # Проекция центра на луч сохраняет направление вектора весов.
        scale = max(0.0, dot((cx, cy), weight))
        scaled[index] = (scale * weight[0], scale * weight[1])
    return labels, centroids, scaled, error / len(points)


def save_plot(path, title, points, labels=None, arrows=None, centres=None, normalized=False):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1000, 820), "white")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype("arial.ttf", 18)
    bold = ImageFont.truetype("arialbd.ttf", 25)
    small = ImageFont.truetype("arial.ttf", 15)
    colors = ["#0069a6", "#bd3d19", "#1b7544", "#7844a8"]
    limit = 1.2 if normalized else 25
    step = 0.5 if normalized else 5
    size = 620

    def pixel(x, y):
        return 500 + x * size / (2 * limit), 390 - y * size / (2 * limit)

    draw.text((500, 32), title, font=bold, fill="#17212b", anchor="mt")
    for i in range(-int(limit / step), int(limit / step) + 1):
        tick = i * step
        color = "#748391" if i == 0 else "#e1e7eb"
        draw.line([pixel(tick, -limit), pixel(tick, limit)], fill=color, width=2)
        draw.line([pixel(-limit, tick), pixel(limit, tick)], fill=color, width=2)
        draw.text((pixel(tick, 0)[0], 713), f"{tick:g}", font=small, fill="#17212b", anchor="mt")
        draw.text((177, pixel(0, tick)[1]), f"{tick:g}", font=small, fill="#17212b", anchor="rm")
    draw.text((500, 742), "Признак x1", font=regular, fill="#17212b", anchor="mt")
    draw.text((45, 390), "x2", font=regular, fill="#17212b")
    if arrows:
        ox, oy = pixel(0, 0)
        for index, end in arrows.items():
            ex, ey = pixel(*end)
            color = colors[index]
            draw.line((ox, oy, ex, ey), fill=color, width=3)
            if (ex, ey) != (ox, oy):
                ux, uy = normalize((ex - ox, ey - oy))
                draw.polygon([(ex, ey), (ex - 14 * ux - 5 * uy, ey - 14 * uy + 5 * ux),
                              (ex - 14 * ux + 5 * uy, ey - 14 * uy - 5 * ux)], fill=color)
                draw.text((ex + 10 * ux, ey + 10 * uy), f"W{index + 1}", font=small,
                          fill=color, anchor="mm", stroke_width=2, stroke_fill="white")
    for index, point in enumerate(points):
        x, y = pixel(*point)
        color = "#234b66" if labels is None else colors[labels[index]]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white")
    if centres:
        for index, point in centres.items():
            x, y = pixel(*point)
            color = colors[index]
            draw.line((x - 7, y - 7, x + 7, y + 7), fill=color, width=3)
            draw.line((x - 7, y + 7, x + 7, y - 7), fill=color, width=3)
    if arrows:
        caption = "Точки — объекты. Стрелки — веса."
        if centres:
            caption += " Кресты — центры кластеров."
        draw.text((500, 782), caption, font=regular, fill="#17212b", anchor="mt")
    image.save(path)


def main():
    folder = Path(__file__).resolve().parent / "результаты_лр4"
    folder.mkdir(exist_ok=True)
    points = [normalize(point) for point in POINTS]
    rng = random.Random(SEED)
    initial_weights = []
    for _ in range(NEURONS):
        angle = rng.uniform(0, math.tau)
        initial_weights.append(normalize((math.cos(angle), math.sin(angle))))
    orders = []
    for _ in range(MAX_EPOCHS):
        order = list(range(len(points)))
        rng.shuffle(order)
        orders.append(order)

    print("ЛАБОРАТОРНАЯ РАБОТА 4. КАРТА КОХОНЕНА. ВАРИАНТ 14")
    print(f"Объектов в таблице — {len(POINTS)}. По исходному графику выбрано {NEURONS} нейрона.")

    print("\n1. ДАННЫЕ ВАРИАНТА")
    print(f"{'№':>3} {'x1':>7} {'x2':>7} {'x1 после норм.':>15} {'x2 после норм.':>15}")
    for index, (original, normalized) in enumerate(zip(POINTS, points), 1):
        x, y = original
        nx, ny = normalized
        print(f"{index:>3} {x:>7} {y:>7} {nx:>15.4f} {ny:>15.4f}")

    print("\n2. УСЛОВИЯ ОБУЧЕНИЯ")
    print(f"Эпох для графиков и замера времени — {EPOCHS}.")
    print(f"Предел проверки сходимости — {MAX_EPOCHS} эпох.")
    print(f"Допустимый сдвиг веса — {STABILITY_LIMIT}. Устойчивых эпох подряд — {STABLE_EPOCHS}.")
    print(f"Скорость обучения — от {LEARNING_START} до {LEARNING_END}.")
    print(f"Число замеров времени для каждого режима — {REPEATS}.")
    print(f"Начальное число генератора — {SEED}.")
    print("Начальные веса нейронов")
    for index, (x, y) in enumerate(initial_weights, 1):
        print(f"  W{index} = ({x:.4f}, {y:.4f})")

    print("\n3. РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")

    results, times = benchmark(points, initial_weights, orders[:EPOCHS])
    save_plot(folder / "01_исходные_точки.png", "Вариант 14. Исходные точки", POINTS)
    comparison = []
    for mode, label, prefix in [(False, "Победитель забирает всё", "wta"),
                                (True, "Обучение с утомляемостью", "fatigue")]:
        weights, wins, _ = results[mode]
        _, _, convergence_epoch = train(points, initial_weights, orders, mode, check_convergence=True)
        convergence = str(convergence_epoch) if convergence_epoch is not None else f"Нет за {MAX_EPOCHS}"
        labels, centres, scaled, error = summarize(points, weights)
        comparison.append((median(times[mode]), len(centres), error, convergence))
        print(f"\n{label}")
        for i, (wx, wy) in enumerate(weights):
            members = [str(j + 1) for j, assigned in enumerate(labels) if assigned == i]
            print(f"  Нейрон W{i + 1}. Итоговый вес ({wx:.4f}, {wy:.4f}).")
            print(f"    Побед за {EPOCHS} эпох — {wins[i]}.")
            if wins[i] == 0:
                print("    Мёртвый нейрон. Ни одной победы, вес сохранил начальное значение.")
            if not members:
                print("    Объекты не назначены. Центр и масштабированная стрелка отсутствуют.")
                continue
            print(f"    Номера объектов ({len(members)}) — {', '.join(members)}.")
            cx, cy = centres[i]
            sx, sy = scaled[i]
            print(f"    Центр кластера ({cx:.3f}, {cy:.3f}). Конец стрелки ({sx:.3f}, {sy:.3f}).")

        arrows = dict(enumerate(weights))
        save_plot(folder / f"{prefix}_нормированные.png", label + ". Нормированные точки",
                  points, labels, arrows, normalized=True)
        save_plot(folder / f"{prefix}_исходные.png", label + ". Исходный масштаб",
                  POINTS, labels, scaled, centres)
        with open(folder / f"{prefix}_кластеры.csv", "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["Объект", "x1", "x2", "Нейрон"])
            for index, ((x, y), group) in enumerate(zip(POINTS, labels), 1):
                writer.writerow([index, x, y, group + 1])
    print("\n4. СРАВНЕНИЕ ДВУХ РЕЖИМОВ")
    classic_time, classic_count, classic_error, classic_epochs = comparison[0]
    fatigue_time, fatigue_count, fatigue_error, fatigue_epochs = comparison[1]
    print(f"{'Показатель':<45} {'Обычный режим':>16} {'С утомляемостью':>17}")
    print(f"{'Эпох до сходимости':<45} {classic_epochs:>16} {fatigue_epochs:>17}")
    print(f"{'Нейроны с объектами после 80 эпох':<45} {classic_count:>16} {fatigue_count:>17}")
    print(f"{'Среднее расстояние (норм. пространство)':<45} {classic_error:>16.4f} {fatigue_error:>17.4f}")
    print(f"{'Медиана времени за 80 эпох, мс':<45} {classic_time:>16.3f} {fatigue_time:>17.3f}")
    print(f"\nГрафики и таблицы сохранены в папке {folder}")



if __name__ == "__main__":
    main()



