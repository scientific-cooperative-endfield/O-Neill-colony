//! egui overlay: HUD, graphics settings, colony controls, balls, teleports.

use crate::physics::BALL_TYPES;
use crate::settings::{AaMode, Graphics, Preset, Tonemap};

pub struct Sim {
    pub tod: f32,
    pub day_minutes: f32,
    pub paused: bool,
    pub time: f32,
    pub spin: f32,
    pub g_floor: f32,
    pub coriolis: bool,
    pub wind_dir: f32,
    pub wind: f32,
    pub debug_field: bool,
    pub time_scale: f32,
    /// cloud cover of the 1.5 km deck (0 = off)
    pub clouds: f32,
}

impl Default for Sim {
    fn default() -> Self {
        Sim {
            tod: 0.36,
            day_minutes: 12.0,
            paused: false,
            time: 0.0,
            spin: 0.0,
            g_floor: 1.0,
            coriolis: true,
            wind_dir: 0.6,
            wind: 0.6,
            debug_field: false,
            time_scale: 1.0,
            clouds: 0.32,
        }
    }
}

pub struct UiState {
    pub show_settings: bool,
    pub show_world: bool,
    pub show_balls: bool,
    pub show_teleports: bool,
    pub show_help: bool,
    pub hidden: bool,
    pub ball_type: usize,
    pub throw_speed: f32,
    pub always_trajectory: bool,
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            show_settings: false,
            show_world: false,
            show_balls: false,
            show_teleports: false,
            show_help: true,
            hidden: false,
            ball_type: 0,
            throw_speed: 18.0,
            always_trajectory: false,
        }
    }
}

pub struct Hud {
    pub fps: f32,
    pub ms: f32,
    pub g: f32,
    pub altitude: f32,
    pub axis_dist: f32,
    pub speed: f32,
    pub valley: &'static str,
    pub blades: u32,
    pub objects: u32,
    pub balls: usize,
    pub fly: bool,
    pub riding: bool,
    pub captured: bool,
    pub adapter: String,
    pub deflection: Option<f32>,
    pub n_buildings: usize,
    pub n_trees: usize,
}

pub enum Action {
    Teleport(usize),
    Rain,
    Clear,
    Spawn,
    Ride,
}

pub fn draw(ctx: &egui::Context, ui: &mut UiState, sim: &mut Sim, gfx: &mut Graphics, hud: &Hud, teleports: &[&str]) -> Vec<Action> {
    let mut actions = Vec::new();
    if ui.hidden {
        return actions;
    }
    let frame = egui::Frame::window(&ctx.style()).fill(egui::Color32::from_rgba_unmultiplied(12, 16, 24, 210));

    // ---------------------------------------------------------------- HUD --
    egui::Area::new(egui::Id::new("hud")).anchor(egui::Align2::LEFT_TOP, [8.0, 8.0]).interactable(false).show(ctx, |u| {
        frame.show(u, |u| {
            u.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
            let h = (sim.tod * 24.0) as u32;
            let m = ((sim.tod * 24.0).fract() * 60.0) as u32;
            u.label(egui::RichText::new(format!("{:>5.1} FPS  {:>5.2} мс", hud.fps, hud.ms)).color(egui::Color32::from_rgb(140, 255, 160)));
            u.label(format!("{}  {:02}:{:02}{}", hud.valley, h, m, if sim.paused { "  [пауза]" } else { "" }));
            u.label(format!("g = {:.2}   высота {:.0} м   до оси {:.0} м", hud.g, hud.altitude, hud.axis_dist));
            u.label(format!(
                "скорость {:.0} км/ч  {}{}",
                hud.speed * 3.6,
                if hud.fly { "полёт" } else { "пешком" },
                if hud.riding { " · на поезде" } else { "" }
            ));
            u.label(format!("трава {}k  объекты {}  мячи {}", (hud.blades + 500) / 1000, hud.objects, hud.balls));
            u.label(format!("мяч [1-8]: {}", BALL_TYPES[ui.ball_type].name));
            if let Some(d) = hud.deflection {
                u.label(egui::RichText::new(format!("Кориолис уводит бросок на {:.1} м", d)).color(egui::Color32::from_rgb(255, 210, 120)));
            }
            if hud.captured {
                u.label(egui::RichText::new("Tab — меню и курсор").weak());
            }
        });
    });
    if hud.captured {
        // Cross-hair.
        let c = ctx.screen_rect().center();
        let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("xhair")));
        let s = egui::Stroke::new(1.5_f32, egui::Color32::from_white_alpha(200));
        p.line_segment([c - egui::vec2(7.0, 0.0), c + egui::vec2(7.0, 0.0)], s);
        p.line_segment([c - egui::vec2(0.0, 7.0), c + egui::vec2(0.0, 7.0)], s);
        return actions;
    }

    // --------------------------------------------------------- menu bar ----
    egui::Area::new(egui::Id::new("menu")).anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0]).show(ctx, |u| {
        frame.show(u, |u| {
            u.horizontal(|u| {
                u.toggle_value(&mut ui.show_settings, "⚙ Графика");
                u.toggle_value(&mut ui.show_world, "🌍 Колония");
                u.toggle_value(&mut ui.show_balls, "⚽ Мячи");
                u.toggle_value(&mut ui.show_teleports, "✈ Телепорты");
                u.toggle_value(&mut ui.show_help, "❓ Справка");
            });
        });
    });

    // --------------------------------------------------------- graphics ----
    let mut open = ui.show_settings;
    egui::Window::new("Настройки графики").open(&mut open).default_pos([980.0, 60.0]).default_width(320.0).frame(frame).show(ctx, |u| {
        let before = gfx.clone();
        u.horizontal_wrapped(|u| {
            u.label("Пресет:");
            for p in Preset::ALL {
                if u.selectable_label(gfx.preset == p, p.label()).clicked() {
                    let vs = gfx.vsync;
                    let cap = gfx.fps_cap;
                    *gfx = Graphics::preset(p);
                    gfx.vsync = vs;
                    gfx.fps_cap = cap;
                }
            }
        });
        u.label(egui::RichText::new(format!("Текущий: {}", gfx.preset.label())).weak());
        u.separator();
        let mut changed = false;
        egui::Grid::new("gfx").num_columns(2).spacing([10.0, 6.0]).show(u, |u| {
            u.label("Масштаб рендера");
            changed |= u.add(egui::Slider::new(&mut gfx.render_scale, 0.33..=1.0).fixed_decimals(2)).changed();
            u.end_row();
            u.label("Сглаживание");
            egui::ComboBox::from_id_salt("aa").selected_text(gfx.aa.label()).show_ui(u, |u| {
                for a in AaMode::ALL {
                    changed |= u.selectable_value(&mut gfx.aa, a, a.label()).changed();
                }
            });
            u.end_row();
            u.label("Тени");
            let lbl = if gfx.shadow_res == 0 { "Выкл".to_string() } else { gfx.shadow_res.to_string() };
            egui::ComboBox::from_id_salt("sh").selected_text(lbl).show_ui(u, |u| {
                for r in [0u32, 1024, 2048, 4096] {
                    changed |= u.selectable_value(&mut gfx.shadow_res, r, if r == 0 { "Выкл".to_string() } else { r.to_string() }).changed();
                }
            });
            u.end_row();
            u.label("Дальность теней");
            changed |= u.add(egui::Slider::new(&mut gfx.shadow_dist, 40.0..=400.0).suffix(" м")).changed();
            u.end_row();
            u.label("Шаг травы");
            changed |= u.add(egui::Slider::new(&mut gfx.grass_spacing, 0.05..=0.25).fixed_decimals(3).suffix(" м")).changed();
            u.end_row();
            u.label("Радиус травы");
            changed |= u.add(egui::Slider::new(&mut gfx.grass_radius, 0.0..=120.0).suffix(" м")).changed();
            u.end_row();
            u.label("Детальная трава до");
            changed |= u.add(egui::Slider::new(&mut gfx.grass_lod, 4.0..=40.0).suffix(" м")).changed();
            u.end_row();
            u.label("Дальность деревьев");
            changed |= u.add(egui::Slider::new(&mut gfx.tree_dist, 100.0..=1500.0).suffix(" м")).changed();
            u.end_row();
            u.label("Детализация рельефа");
            changed |= u.add(egui::Slider::new(&mut gfx.terrain_detail, 0.25..=2.5)).changed();
            u.end_row();
            u.label("Аэромобили");
            changed |= u.add(egui::Slider::new(&mut gfx.aircars, 0..=3000)).changed();
            u.end_row();
            u.label("Bloom");
            u.horizontal(|u| {
                changed |= u.checkbox(&mut gfx.bloom, "").changed();
                changed |= u.add(egui::Slider::new(&mut gfx.bloom_strength, 0.0..=0.3)).changed();
            });
            u.end_row();
            u.label("Тонмаппинг");
            u.horizontal(|u| {
                changed |= u.selectable_value(&mut gfx.tonemap, Tonemap::AgX, "AgX").changed();
                changed |= u.selectable_value(&mut gfx.tonemap, Tonemap::Aces, "ACES").changed();
            });
            u.end_row();
            u.label("Экспозиция (EV)");
            changed |= u.add(egui::Slider::new(&mut gfx.exposure_bias, -3.0..=3.0)).changed();
            u.end_row();
            u.label("Насыщенность");
            changed |= u.add(egui::Slider::new(&mut gfx.saturation, 0.0..=1.6)).changed();
            u.end_row();
            u.label("Резкость (CAS)");
            changed |= u.add(egui::Slider::new(&mut gfx.sharpen, 0.0..=1.0)).changed();
            u.end_row();
            u.label("Виньетка");
            changed |= u.add(egui::Slider::new(&mut gfx.vignette, 0.0..=0.8)).changed();
            u.end_row();
            u.label("Дымка");
            changed |= u.add(egui::Slider::new(&mut gfx.fog, 0.0..=3.0)).changed();
            u.end_row();
            u.label("V-Sync");
            changed |= u.checkbox(&mut gfx.vsync, "").changed();
            u.end_row();
            u.label("Лимит FPS");
            changed |=
                u.add(egui::Slider::new(&mut gfx.fps_cap, 0..=240).custom_formatter(|v, _| if v == 0.0 { "нет".into() } else { format!("{v:.0}") })).changed();
            u.end_row();
        });
        if changed && *gfx != before {
            gfx.preset = Preset::Custom;
        }
        u.separator();
        u.label(egui::RichText::new(format!("GPU: {}", hud.adapter)).weak());
        u.label(egui::RichText::new("Совет: на слабой видеокарте — «Чайник» или «Низкие», масштаб 0.5–0.7 + резкость.").weak());
    });
    ui.show_settings = open;

    // ----------------------------------------------------------- colony ----
    let mut open = ui.show_world;
    egui::Window::new("Колония О'Нила").open(&mut open).default_pos([980.0, 60.0]).default_width(300.0).frame(frame).show(ctx, |u| {
        u.label("Остров III: R = 4 км, длина 32 км (канон Island Three)");
        u.separator();
        u.horizontal(|u| {
            u.label("Время суток");
            u.add(egui::Slider::new(&mut sim.tod, 0.0..=1.0).custom_formatter(|v, _| {
                let h = (v * 24.0) as u32;
                format!("{:02}:{:02}", h, ((v * 24.0).fract() * 60.0) as u32)
            }));
        });
        u.horizontal(|u| {
            u.label("Длина суток");
            u.add(egui::Slider::new(&mut sim.day_minutes, 1.0..=60.0).suffix(" мин"));
        });
        u.horizontal(|u| {
            u.checkbox(&mut sim.paused, "Пауза (P)");
            u.label("Скорость");
            u.add(egui::Slider::new(&mut sim.time_scale, 0.1..=3.0));
        });
        u.separator();
        u.horizontal(|u| {
            u.label("Гравитация у пола");
            u.add(egui::Slider::new(&mut sim.g_floor, 0.0..=1.5).suffix(" g"));
        });
        let omega = crate::colony::omega_for(sim.g_floor);
        u.label(
            egui::RichText::new(format!("ω = {:.4} рад/с · оборот за {:.0} с", omega, if omega > 0.0 { std::f32::consts::TAU / omega } else { f32::INFINITY }))
                .weak(),
        );
        u.checkbox(&mut sim.coriolis, "Сила Кориолиса");
        u.separator();
        u.horizontal(|u| {
            u.label("Ветер");
            u.add(egui::Slider::new(&mut sim.wind, 0.0..=2.0));
        });
        u.horizontal(|u| {
            u.label("Направление");
            u.add(egui::Slider::new(&mut sim.wind_dir, -std::f32::consts::PI..=std::f32::consts::PI));
        });
        u.horizontal(|u| {
            u.label("Облака (1,5 км)");
            u.add(egui::Slider::new(&mut sim.clouds, 0.0..=0.8));
        });
        u.checkbox(&mut sim.debug_field, "Показать поле динамики травы (V)");
        u.label(egui::RichText::new("B — режим «чёрная дыра» (Sgr A*)").weak());
        u.separator();
        u.label(egui::RichText::new(format!("Зданий: {} · деревьев: {}", hud.n_buildings, hud.n_trees)).weak());
    });
    ui.show_world = open;

    // ------------------------------------------------------------ balls ----
    let mut open = ui.show_balls;
    egui::Window::new("Мячи").open(&mut open).default_pos([8.0, 200.0]).default_width(250.0).frame(frame).show(ctx, |u| {
        for (i, t) in BALL_TYPES.iter().enumerate() {
            u.radio_value(&mut ui.ball_type, i, format!("{} · {}", i + 1, t.name)).on_hover_text(format!(
                "r = {:.2} м, m = {} кг, плотность {:.0} кг/м³",
                t.radius,
                t.mass,
                t.density()
            ));
        }
        u.horizontal(|u| {
            u.label("Сила броска");
            u.add(egui::Slider::new(&mut ui.throw_speed, 2.0..=60.0).suffix(" м/с"));
        });
        u.checkbox(&mut ui.always_trajectory, "Всегда показывать траекторию");
        u.horizontal(|u| {
            if u.button("Бросить").clicked() {
                actions.push(Action::Spawn);
            }
            if u.button("☔ Дождь с оси").clicked() {
                actions.push(Action::Rain);
            }
            if u.button("Убрать все").clicked() {
                actions.push(Action::Clear);
            }
        });
        u.label(egui::RichText::new("Зелёная линия — реальный полёт во вращающейся колонии, красная — как было бы на Земле.").weak());
    });
    ui.show_balls = open;

    // -------------------------------------------------------- teleports ----
    let mut open = ui.show_teleports;
    egui::Window::new("Телепорты").open(&mut open).default_pos([700.0, 60.0]).default_width(250.0).frame(frame).show(ctx, |u| {
        for (i, name) in teleports.iter().enumerate() {
            if u.button(*name).clicked() {
                actions.push(Action::Teleport(i));
            }
        }
        if u.button("🚄 Сесть на ближайший маглев (T)").clicked() {
            actions.push(Action::Ride);
        }
    });
    ui.show_teleports = open;

    // ------------------------------------------------------------- help ----
    let mut open = ui.show_help;
    egui::Window::new("Управление").open(&mut open).default_pos([300.0, 90.0]).default_width(380.0).frame(frame).show(ctx, |u| {
        u.label("Клик по сцене или Tab — захват мыши, Tab/Esc — вернуть курсор.");
        egui::Grid::new("help").num_columns(2).show(u, |u| {
            for (k, d) in [
                ("WASD / Shift", "ходьба / бег"),
                ("Пробел", "прыжок (в полёте — вверх)"),
                ("C / Ctrl", "вниз в режиме полёта"),
                ("F", "полёт вкл/выкл"),
                ("Колесо", "скорость полёта"),
                ("ЛКМ", "бросить мяч"),
                ("ПКМ (держать)", "прицел с траекторией"),
                ("1–8", "тип мяча"),
                ("G", "пнуть ближайший мяч"),
                ("R", "дождь мячей с оси"),
                ("T", "сесть на маглев / сойти"),
                ("P", "пауза времени"),
                ("V", "поле динамики травы"),
                ("F1", "скрыть интерфейс"),
            ] {
                u.label(egui::RichText::new(k).strong());
                u.label(d);
                u.end_row();
            }
        });
        u.separator();
        u.label("Изюминка: в колонии нет гравитации — только вращение. Брошенный мяч летит по прямой в инерциальной системе, а для вас уходит вбок (Кориолис). Гелиевый шар всплывает к оси, а у оси наступает невесомость.");
    });
    ui.show_help = open;
    actions
}
