# O’Neill Colony — «Остров Три»

Симулятор вращающейся космической колонии на **Rust, wgpu и WGSL**:
процедурный мир, физика Кориолиса, режим чёрной дыры с разрушением станции
и схватки с ботами. Интерфейс построен на egui; модели оружия и оперативников
созданы скриптами Blender и включены в репозиторий.

![Колония и бой от первого лица](screenshots/v10_first_person.png)

## Быстрый запуск

Нужны актуальный стабильный Rust с Cargo, системный C-компилятор и графический
драйвер с поддержкой Vulkan. Основная среда разработки — Linux (Fedora / Ultramarine).
Запускайте команды из корня проекта: сценарии и кеши используют относительные пути.

```bash
cargo build --release --locked --workspace --bins
cargo run --release --locked --bin oneill_colony -- --preset medium
```

Для слабой видеокарты начните с `--preset teapot`; также доступны `low`, `high`, `ultra`.
На Fedora системные зависимости можно установить так:

```bash
sudo dnf install gcc pkgconf-pkg-config wayland-devel libxkbcommon-devel vulkan-loader vulkan-tools
```

Драйвер видеокарты устанавливается отдельно под ваше оборудование.
`vulkaninfo --summary` позволяет проверить доступность Vulkan.
Blender нужен только для изменения моделей; готовые `.omdl` уже включены в проект.

## Режимы

```bash
# Колония: вы и восемь ботов
cargo run --release --locked --bin oneill_colony -- --skirmish 8

# Наблюдение за ботами
cargo run --release --locked --bin oneill_colony -- --skirmish 8 --spectate

# Колония, падающая в чёрную дыру
cargo run --release --locked --bin oneill_colony -- --blackhole scenarios/colony_sgr.toml

# Отдельный визуализатор чёрной дыры
cargo run --release --locked --bin blackhole

# Экспорт кадра без окна (требует Vulkan, включая программный драйвер)
cargo run --release --locked --bin blackhole -- --headless --scenario scenarios/edge_on.toml --frames 1 --out out

# Симуляция боя без графики с проверкой повторяемости
cargo run --release --locked -p ai --bin botmatch -- --minutes 10 --bots 8 --seed 1 --verify 3600
```

Карта HDR-неба генерируется при первом запуске и кешируется в `assets/sky_*.bin`.
Для предварительной генерации: `cargo run --release --locked --bin gen_sky -- 1024`.

## Управление

| Клавиши | Действие |
|---|---|
| WASD, мышь, Shift | Движение, взгляд, бег |
| Space, C / Ctrl | Прыжок / движение вверх, вниз |
| Tab | Захват мыши / меню; в бою — таблица счёта |
| F | Полёт; в бою — ранец |
| B | Режим чёрной дыры |
| K | Начать / завершить схватку |
| ЛКМ / ПКМ | Бросок / траектория; в бою — огонь / прицеливание |
| 1–3, R (в бою) | Оружие, перезарядка |
| Q / E, N (в бою) | Устройство / режим, нейро-проекция траектории |
| F12 | Скриншот в корне проекта |

## Структура

| Каталог | Назначение |
|---|---|
| `src/` | Приложение колонии, рендер, интерфейс, интеграция боя |
| `src/bh/`, `src/bin/blackhole/` | Физика и визуализация чёрной дыры |
| `crates/relativity/` | Метрика, геодезические, гравитационный потенциал |
| `crates/structure/` | Структура станции, разрушение и разгерметизация |
| `crates/oneill_world/` | Генерация мира и физика колонии |
| `crates/gameplay/`, `crates/ai/` | Игровое ядро, боты и headless botmatch |
| `shaders/` | WGSL-шейдеры |
| `scenarios/` | TOML-сценарии |
| `assets/`, `tools/` | Модели, данные разрушения и генераторы |
| `screenshots/` | Иллюстрации проекта |

## Проверки

```bash
cargo test --locked --workspace
cargo check --locked --workspace --all-targets
cargo fmt --all -- --check
```

Тесты физики, детерминизма, ботов и валидации WGSL запускаются без GPU.
GitHub Actions выполняет сборку всех бинарников и тесты workspace на Linux.
Для проверки форматирования нужен компонент `rustfmt`.

Проект экспериментальный. Производительность рендера зависит от GPU;
детерминизм симуляции проверяется в пределах одной сборки и платформы.

[История разработки](docs/CHANGELOG.md) · [Технические решения](DECISIONS.md) ·
[Отправка на GitHub](docs/GITHUB.md)

## Лицензия

Лицензия проекта пока не выбрана. Перед распространением с разрешением на
использование и модификацию добавьте подходящий `LICENSE` и сведения в Cargo.toml.
