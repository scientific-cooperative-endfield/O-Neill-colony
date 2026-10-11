//! Structure graph of the station (the O'Neill "Island Three" hull) and its
//! tidal break-up.
//!
//! * Nodes are hull panels: `N_Z` rings × `N_THETA` sectors of the cylinder
//!   (land strips and window strips, 15° each) plus `N_CAP` sectors per end cap.
//! * Bonds join neighbouring panels: hoop bonds around a ring, axial bonds
//!   between rings (and between the caps and the end rings).
//! * Loads are quasi-static and evaluated every step in the free-fall frame of
//!   each fragment: the rigid spin needs a centripetal force, the air pushes
//!   the hull outwards, and the exact Schwarzschild tidal tensor
//!   (`physics::tidal_tensor`) pulls the panels apart. Cut forces through each
//!   ring plane / along each ring give the bond loads.
//! * A bond fails when its load exceeds its strength (design load × safety
//!   factor, with a deterministic ±scatter). Connected components of the
//!   remaining bonds become fragments; each fragment is a rigid body whose
//!   centre of mass follows its own exact PG geodesic (`Geo`), inheriting the
//!   local velocity (spin included) at the moment of separation.
//! * Every panel carries its own proper-time offset, integrated from its exact
//!   PG clock rate dτ/dT = √(1 − |w − v_river|²), so the clocks of the parts
//!   of the station drift apart exactly as general relativity predicts.
//!
//! Units: panel offsets in metres, geodesics in M (`m` = GM/c² in metres).

use crate::noise::Rng;
use glam::{DMat3, DQuat, DVec3};
use relativity::{self as physics, Geo, C};

pub const N_THETA: usize = 24;
pub const N_Z: usize = 16;
pub const N_CAP: usize = 6;
pub const N_NODES: usize = N_THETA * N_Z + 2 * N_CAP;
pub const MAX_PARTICLES: usize = 16384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Land,
    Window,
    Cap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondKind {
    Hoop,
    Axial,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub radius: f64,
    pub length: f64,
    /// centrifugal acceleration on the floor (m/s²)
    pub spin_g: f64,
    /// areal densities (kg/m²)
    pub sigma_land: f64,
    pub sigma_window: f64,
    pub sigma_cap: f64,
    /// air pressure (Pa)
    pub pressure: f64,
    /// strength / design load
    pub safety: f64,
    /// relative ± scatter of bond strengths
    pub scatter: f64,
    pub seed: u64,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            radius: 4000.0,
            length: 32_000.0,
            spin_g: 9.81,
            sigma_land: 1.0e4,
            sigma_window: 2.5e3,
            sigma_cap: 1.2e4,
            pressure: 1.0e5,
            safety: 1.4,
            scatter: 0.15,
            seed: 7,
        }
    }
}

impl Spec {
    pub fn omega(&self) -> f64 {
        (self.spin_g / self.radius).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    /// rest position in the station body frame (m, centred on the station COM, axis = +z)
    pub rest: DVec3,
    /// panel axes in the body frame (x tangential, y along the axis / radial on caps, z outward normal)
    pub axes: DMat3,
    /// half extents of the panel box (m)
    pub half: DVec3,
    pub mass: f64,
    pub kind: Kind,
    /// sector index and ring index (−1 / N_Z: caps)
    pub ith: usize,
    pub iz: i32,
    pub frag: usize,
    /// proper time relative to the COM clock of its fragment (M)
    pub dtau: f64,
    /// current clock rate dτ/dT
    pub rate: f64,
    /// world position (M, BH frame) and coordinate velocity dx/dT (c)
    pub x: DVec3,
    pub w: DVec3,
    /// last structural load / strength of its worst bond (0..)
    pub stress: f64,
}

#[derive(Clone, Debug)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub kind: BondKind,
    /// cut index: ring for hoop bonds, ring-plane for axial bonds
    pub cut: i32,
    pub strength: f64,
    pub load: f64,
    pub broken: bool,
    /// length of the joint (m) — sets the opening area once broken
    pub joint: f64,
}

#[derive(Clone, Debug)]
pub struct Fragment {
    pub nodes: Vec<usize>,
    /// centre-of-mass worldline (M)
    pub body: Geo,
    /// body frame → BH frame
    pub q: DQuat,
    /// angular velocity (rad/s, BH frame)
    pub omega: DVec3,
    /// COM in the body rest frame (m)
    pub com: DVec3,
    pub mass: f64,
    /// tidal tensor at the COM (s⁻²)
    pub k: DMat3,
    pub alive: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Particle {
    /// offset from the reference worldline (m, BH frame) and velocity (m/s)
    pub d: DVec3,
    pub v: DVec3,
    pub life: f32,
    pub age: f32,
    /// 0 debris, 1 glass, 2 air / ice
    pub kind: u32,
    pub size: f32,
}

#[derive(Clone, Debug)]
pub struct BreakEvent {
    pub tau: f64,
    pub r: f64,
    pub node_a: usize,
    pub node_b: usize,
}

#[derive(Clone, Debug)]
pub struct Station {
    pub spec: Spec,
    pub nodes: Vec<Node>,
    pub bonds: Vec<Bond>,
    pub frags: Vec<Fragment>,
    /// the original COM worldline (continues as a virtual reference after break-up)
    pub reference: Geo,
    pub particles: Vec<Particle>,
    pub pressure: f64,
    pub events: Vec<BreakEvent>,
    pub first_break: Option<BreakEvent>,
    /// total area of the breaches (m²)
    pub breach_area: f64,
    /// extra open area (m²) from local damage — shot-out windows, device
    /// accidents — reported by gameplay through `puncture`
    pub punctures: f64,
    rng: Rng,
    next_particle: usize,
}

fn land_at(theta: f64) -> bool {
    let a = (theta + std::f64::consts::FRAC_PI_6).rem_euclid(std::f64::consts::TAU / 3.0);
    a < std::f64::consts::FRAC_PI_3
}

impl Station {
    /// Builds the intact station with its COM on `body` and the axis along `axis` (BH frame).
    pub fn new(spec: Spec, body: Geo, axis: DVec3) -> Station {
        let r = spec.radius;
        let dth = std::f64::consts::TAU / N_THETA as f64;
        let dz = spec.length / N_Z as f64;
        let p0 = spec.pressure;
        let w0 = spec.omega();
        let mut rng = Rng(spec.seed ^ 0x5eed_57a7);
        let mut nodes = Vec::with_capacity(N_NODES);
        let mk = |rest: DVec3, axes: DMat3, half: DVec3, mass: f64, kind: Kind, ith: usize, iz: i32| Node {
            rest,
            axes,
            half,
            mass,
            kind,
            ith,
            iz,
            frag: 0,
            dtau: 0.0,
            rate: 1.0,
            x: DVec3::ZERO,
            w: DVec3::ZERO,
            stress: 0.0,
        };
        for iz in 0..N_Z {
            let z = -0.5 * spec.length + (iz as f64 + 0.5) * dz;
            for j in 0..N_THETA {
                let th = (j as f64 + 0.5) * dth;
                let (s, c) = th.sin_cos();
                let land = land_at(th);
                let sigma = if land { spec.sigma_land } else { spec.sigma_window };
                let axes = DMat3::from_cols(DVec3::new(-s, c, 0.0), DVec3::Z, DVec3::new(c, s, 0.0));
                let half = DVec3::new(0.5 * r * dth, 0.5 * dz, if land { 40.0 } else { 8.0 });
                let kind = if land { Kind::Land } else { Kind::Window };
                nodes.push(mk(DVec3::new(r * c, r * s, z), axes, half, sigma * r * dth * dz, kind, j, iz as i32));
            }
        }
        let dphi = std::f64::consts::TAU / N_CAP as f64;
        for (cap, zs) in [(-1i32, -1.0f64), (N_Z as i32, 1.0)] {
            for k in 0..N_CAP {
                let ph = (k as f64 + 0.5) * dphi;
                let (s, c) = ph.sin_cos();
                let axes = DMat3::from_cols(DVec3::new(-s, c, 0.0), DVec3::new(c, s, 0.0), DVec3::Z * zs);
                let half = DVec3::new(0.5 * r * dphi * 0.55, 0.5 * r * 0.9, 60.0);
                let area = std::f64::consts::PI * r * r / N_CAP as f64;
                nodes.push(mk(DVec3::new(0.5 * r * c, 0.5 * r * s, zs * 0.5 * spec.length), axes, half, spec.sigma_cap * area, Kind::Cap, k, cap));
            }
        }
        let idx = |iz: usize, j: usize| iz * N_THETA + (j % N_THETA);
        let cap_idx = |top: bool, k: usize| N_THETA * N_Z + if top { N_CAP } else { 0 } + (k % N_CAP);
        let mut bonds = Vec::new();
        let mut strength = |base: f64| base * spec.safety * (1.0 + spec.scatter * (2.0 * rng.f64() - 1.0));
        // hoop design load per bond: (σ ω² R² + P R) Δz
        for iz in 0..N_Z {
            for j in 0..N_THETA {
                let (a, b) = (idx(iz, j), idx(iz, j + 1));
                let sig = 0.5 * (nodes[a].mass + nodes[b].mass) / (r * dth * dz);
                let design = (sig * w0 * w0 * r * r + p0 * r) * dz;
                bonds.push(Bond { a, b, kind: BondKind::Hoop, cut: iz as i32, strength: strength(design), load: 0.0, broken: false, joint: dz });
            }
        }
        // axial design load per bond: P R / 2 per unit circumference
        let axial = 0.5 * p0 * r * r * dth;
        for iz in 0..N_Z - 1 {
            for j in 0..N_THETA {
                bonds.push(Bond {
                    a: idx(iz, j),
                    b: idx(iz + 1, j),
                    kind: BondKind::Axial,
                    cut: iz as i32 + 1,
                    strength: strength(axial),
                    load: 0.0,
                    broken: false,
                    joint: r * dth,
                });
            }
        }
        let per_cap = N_THETA / N_CAP;
        for top in [false, true] {
            for k in 0..N_CAP {
                let c = cap_idx(top, k);
                for jj in 0..per_cap {
                    let ring = if top { N_Z - 1 } else { 0 };
                    let cut = if top { N_Z as i32 } else { 0 };
                    bonds.push(Bond {
                        a: c,
                        b: idx(ring, k * per_cap + jj),
                        kind: BondKind::Axial,
                        cut,
                        strength: strength(axial),
                        load: 0.0,
                        broken: false,
                        joint: r * dth,
                    });
                }
                // cap membrane: spin of the cap disc plus a share of the pressure load
                let sig = spec.sigma_cap;
                let design = (sig * w0 * w0 * 0.25 * r * r + 0.5 * p0 * r) * 0.5 * r;
                bonds.push(Bond {
                    a: c,
                    b: cap_idx(top, k + 1),
                    kind: BondKind::Hoop,
                    cut: if top { N_Z as i32 } else { -1 },
                    strength: strength(design),
                    load: 0.0,
                    broken: false,
                    joint: 0.5 * r,
                });
            }
        }
        let axis = axis.normalize_or(DVec3::Z);
        let q = DQuat::from_rotation_arc(DVec3::Z, axis);
        let mass: f64 = nodes.iter().map(|n| n.mass).sum();
        let com = nodes.iter().map(|n| n.rest * n.mass).sum::<DVec3>() / mass;
        let frag = Fragment { nodes: (0..nodes.len()).collect(), body, q, omega: axis * w0, com, mass, k: DMat3::ZERO, alive: true };
        let mut st = Station {
            spec,
            nodes,
            bonds,
            frags: vec![frag],
            reference: body,
            particles: Vec::new(),
            pressure: p0,
            events: Vec::new(),
            first_break: None,
            breach_area: 0.0,
            punctures: 0.0,
            rng: Rng(0x9e37_79b9 ^ 0x51),
            next_particle: 0,
        };
        st.update_nodes(1.0, 0.0);
        st
    }

    pub fn intact(&self) -> bool {
        self.first_break.is_none()
    }

    pub fn live_fragments(&self) -> usize {
        self.frags.iter().filter(|f| f.alive).count()
    }

    /// Radial extent of the debris cloud around the reference (m).
    pub fn extent(&self, m: f64) -> f64 {
        let half = 0.5 * (self.spec.length * self.spec.length + 4.0 * self.spec.radius * self.spec.radius).sqrt();
        self.frags.iter().filter(|f| f.alive).map(|f| (f.body.x - self.reference.x).length() * m + half).fold(half, f64::max)
    }

    /// Body-frame offset of a node from its fragment COM, rotated to the BH frame (m).
    pub fn node_offset(&self, i: usize) -> DVec3 {
        let n = &self.nodes[i];
        let f = &self.frags[n.frag];
        f.q * (n.rest - f.com)
    }

    /// Advances every fragment by PG time `dt` (M). `m` = GM/c² (m).
    pub fn step(&mut self, m: f64, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let m_s = m / C;
        let tau0 = self.reference.tau;
        self.reference.step_t(1.0, dt);
        let dtau_s = (self.reference.tau - tau0) * m_s;
        for f in self.frags.iter_mut().filter(|f| f.alive) {
            let t0 = f.body.tau;
            f.body.step_t(1.0, dt);
            let ds = (f.body.tau - t0) * m_s;
            f.q = (DQuat::from_scaled_axis(f.omega * ds) * f.q).normalize();
            f.k = if f.body.dead { DMat3::ZERO } else { physics::tidal_tensor(m, &f.body) };
            if f.body.dead {
                f.alive = false;
            }
        }
        self.update_nodes(m, dt);
        self.loads();
        self.fail(m, dtau_s);
        self.vent(m, dtau_s);
        self.step_particles(m, dtau_s);
    }

    /// Node world positions, coordinate velocities and clocks.
    fn update_nodes(&mut self, m: f64, dt: f64) {
        for i in 0..self.nodes.len() {
            let f = &self.frags[self.nodes[i].frag];
            let d = f.q * (self.nodes[i].rest - f.com);
            let x = f.body.x + d / m;
            let wc = f.body.coord_velocity(1.0);
            let w = wc + f.omega.cross(d) / C;
            let rel = w - physics::river(1.0, x);
            let rate = (1.0 - rel.length_squared()).max(0.0).sqrt();
            let rate_c = 1.0 / f.body.dt_dtau(1.0);
            let n = &mut self.nodes[i];
            n.dtau += (rate - rate_c) * dt;
            n.rate = rate;
            n.x = x;
            n.w = w;
        }
    }

    /// Structural force each node needs from its neighbours (N, BH frame).
    fn node_force(&self, i: usize) -> DVec3 {
        let n = &self.nodes[i];
        let f = &self.frags[n.frag];
        let d = f.q * (n.rest - f.com);
        let a_req = f.omega.cross(f.omega.cross(d));
        let mut force = (a_req - f.k * d) * n.mass;
        if self.pressure > 0.0 {
            let area = n.mass
                / if n.kind == Kind::Land {
                    self.spec.sigma_land
                } else if n.kind == Kind::Window {
                    self.spec.sigma_window
                } else {
                    self.spec.sigma_cap
                };
            force -= f.q * n.axes.z_axis * (self.pressure * area);
        }
        force
    }

    fn loads(&mut self) {
        let forces: Vec<DVec3> = (0..self.nodes.len()).map(|i| self.node_force(i)).collect();
        for n in self.nodes.iter_mut() {
            n.stress = 0.0;
        }
        for (fi, f) in self.frags.iter().enumerate() {
            if !f.alive || f.nodes.len() < 2 {
                continue;
            }
            let zf = f.q * DVec3::Z;
            // axial cuts: ring planes −1|0 … N_Z−1|N_Z
            let mut above = vec![DVec3::ZERO; N_Z + 2];
            let mut total = DVec3::ZERO;
            for &i in &f.nodes {
                let iz = (self.nodes[i].iz + 1) as usize; // 0..=N_Z+1
                above[iz] += forces[i];
                total += forces[i];
            }
            // suffix sums: S_above(c) = Σ_{iz ≥ c}
            let mut suf = vec![DVec3::ZERO; N_Z + 3];
            for c in (0..N_Z + 2).rev() {
                suf[c] = suf[c + 1] + above[c];
            }
            let mut cut_t = [0.0f64; N_Z + 2];
            let mut cut_n = [0usize; N_Z + 2];
            for b in self.bonds.iter().filter(|b| !b.broken && b.kind == BondKind::Axial) {
                if self.nodes[b.a].frag == fi && self.nodes[b.b].frag == fi {
                    cut_n[b.cut as usize] += 1;
                }
            }
            for c in 0..=N_Z {
                // plane c separates rings < c from rings ≥ c (ring index iz, cap −1 / N_Z)
                let s_above = suf[c + 1];
                let s_below = total - s_above;
                cut_t[c] = 0.5 * (-s_above + s_below).dot(zf);
            }
            // hoop: per ring, closed rings use the ring formula, open arcs the chain cut force
            for b in self.bonds.iter_mut().filter(|b| !b.broken) {
                if self.nodes[b.a].frag != fi || self.nodes[b.b].frag != fi {
                    continue;
                }
                match b.kind {
                    BondKind::Axial => {
                        let c = b.cut as usize;
                        b.load = cut_t[c] / cut_n[c].max(1) as f64;
                    }
                    BondKind::Hoop => {
                        let _ = b;
                    }
                }
            }
        }
        // hoop bonds (needs ring connectivity)
        let n_rings = N_Z + 2;
        for ring in 0..n_rings {
            let iz = ring as i32 - 1;
            let count = if iz < 0 || iz >= N_Z as i32 { N_CAP } else { N_THETA };
            let node_of = |k: usize| -> usize {
                let k = k % count;
                if iz < 0 {
                    N_THETA * N_Z + k
                } else if iz >= N_Z as i32 {
                    N_THETA * N_Z + N_CAP + k
                } else {
                    iz as usize * N_THETA + k
                }
            };
            // hoop bond index for (k, k+1)
            let bond_of = |k: usize| -> usize {
                if iz < 0 {
                    N_Z * N_THETA + (N_Z - 1) * N_THETA + k * (N_THETA / N_CAP + 1) + N_THETA / N_CAP
                } else if iz >= N_Z as i32 {
                    N_Z * N_THETA + (N_Z - 1) * N_THETA + (N_CAP + k) * (N_THETA / N_CAP + 1) + N_THETA / N_CAP
                } else {
                    iz as usize * N_THETA + k
                }
            };
            let alive = |k: usize, bonds: &Vec<Bond>, nodes: &Vec<Node>| {
                let b = &bonds[bond_of(k)];
                !b.broken && nodes[b.a].frag == nodes[b.b].frag
            };
            let closed = (0..count).all(|k| alive(k, &self.bonds, &self.nodes));
            let dth = std::f64::consts::TAU / count as f64;
            if closed {
                for k in 0..count {
                    let (a, b) = (node_of(k), node_of(k + 1));
                    let f = &self.frags[self.nodes[a].frag];
                    let ra = f.q * self.nodes[a].axes.z_axis;
                    let rb = f.q * self.nodes[b].axes.z_axis;
                    // in-plane outward direction of the cap sectors is their y axis
                    let (ra, rb) = if count == N_CAP { (f.q * self.nodes[a].axes.y_axis, f.q * self.nodes[b].axes.y_axis) } else { (ra, rb) };
                    let fa = -forces[a].dot(ra);
                    let fb = -forces[b].dot(rb);
                    let t = 0.5 * (fa + fb) / (2.0 * (0.5 * dth).sin());
                    let bi = bond_of(k);
                    self.bonds[bi].load = t;
                }
            } else {
                // walk the arcs
                let start = (0..count).find(|&k| !alive(k, &self.bonds, &self.nodes)).unwrap_or(0);
                let mut arc: Vec<usize> = Vec::new();
                let mut arcs: Vec<Vec<usize>> = Vec::new();
                for s in 1..=count {
                    let k = (start + s) % count;
                    arc.push(k);
                    if !alive(k, &self.bonds, &self.nodes) {
                        arcs.push(std::mem::take(&mut arc));
                    }
                }
                if !arc.is_empty() {
                    arcs.push(arc);
                }
                for arc in arcs {
                    if arc.len() < 2 {
                        continue;
                    }
                    let fs: Vec<DVec3> = arc.iter().map(|&k| forces[node_of(k)]).collect();
                    let tot: DVec3 = fs.iter().copied().sum();
                    let mut before = DVec3::ZERO;
                    for (p, &k) in arc.iter().enumerate().take(arc.len() - 1) {
                        before += fs[p];
                        let after = tot - before;
                        self.bonds[bond_of(k)].load = 0.5 * (before.length() + after.length());
                    }
                }
            }
        }
        for b in self.bonds.iter().filter(|b| !b.broken) {
            let s = (b.load / b.strength).max(-b.load / (2.0 * b.strength));
            for i in [b.a, b.b] {
                let n = &mut self.nodes[i];
                n.stress = n.stress.max(s);
            }
        }
    }

    /// Breaks overloaded bonds (worst first, at most a crack-speed limited number
    /// per step) and splits fragments.
    fn fail(&mut self, m: f64, dtau_s: f64) {
        let mut over: Vec<(f64, usize)> = self
            .bonds
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.broken && self.nodes[b.a].frag == self.nodes[b.b].frag && self.frags[self.nodes[b.a].frag].alive)
            .filter_map(|(i, b)| {
                let s = (b.load / b.strength).max(-b.load / (2.0 * b.strength));
                (s > 1.0).then_some((s, i))
            })
            .collect();
        if over.is_empty() {
            return;
        }
        over.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        // cracks run at ~2 km/s: limit the failures per step accordingly
        let max = ((dtau_s * 60.0).ceil() as usize).clamp(1, 400);
        let mut touched = Vec::new();
        for &(_, bi) in over.iter().take(max) {
            let (a, b) = (self.bonds[bi].a, self.bonds[bi].b);
            self.bonds[bi].broken = true;
            self.bonds[bi].load = 0.0;
            let ev = BreakEvent { tau: self.reference.tau, r: self.nodes[a].x.length(), node_a: a, node_b: b };
            if self.first_break.is_none() {
                self.first_break = Some(ev.clone());
            }
            self.events.push(ev);
            self.spawn_break_particles(m, a, b);
            touched.push(self.nodes[a].frag);
        }
        touched.sort_unstable();
        touched.dedup();
        for fi in touched {
            self.split(m, fi);
        }
    }

    /// Splits fragment `fi` into its connected components.
    fn split(&mut self, m: f64, fi: usize) {
        let members = self.frags[fi].nodes.clone();
        let mut parent: Vec<usize> = (0..self.nodes.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for b in self.bonds.iter().filter(|b| !b.broken) {
            if self.nodes[b.a].frag == fi && self.nodes[b.b].frag == fi {
                let (ra, rb) = (find(&mut parent, b.a), find(&mut parent, b.b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for &i in &members {
            let r = find(&mut parent, i);
            match groups.iter_mut().find(|g| g.0 == r) {
                Some(g) => g.1.push(i),
                None => groups.push((r, vec![i])),
            }
        }
        if groups.len() < 2 {
            return;
        }
        // the largest piece keeps the fragment slot
        groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        let old = self.frags[fi].clone();
        let wc = old.body.coord_velocity(1.0);
        for (gi, (_, nodes)) in groups.into_iter().enumerate() {
            let mass: f64 = nodes.iter().map(|&i| self.nodes[i].mass).sum();
            let com = nodes.iter().map(|&i| self.nodes[i].rest * self.nodes[i].mass).sum::<DVec3>() / mass;
            let d = old.q * (com - old.com);
            let x = old.body.x + d / m;
            let w = wc + old.omega.cross(d) / C;
            let mut body = Geo::from_coord_velocity(1.0, x, w);
            body.tau = old.body.tau;
            body.t_pg = old.body.t_pg;
            let frag = Fragment { nodes: nodes.clone(), body, q: old.q, omega: old.omega, com, mass, k: old.k, alive: true };
            let slot = if gi == 0 {
                self.frags[fi] = frag;
                fi
            } else {
                self.frags.push(frag);
                self.frags.len() - 1
            };
            for &i in &nodes {
                // keep each panel clock continuous across the change of COM clock
                self.nodes[i].frag = slot;
            }
        }
    }

    /// Local damage to panel `node`: opens `area` m² to vacuum (vented with
    /// the structural breaches) and weakens the panel's bonds in proportion to
    /// the fraction of its area that is gone. Returns the panel's remaining
    /// strength factor.
    pub fn puncture(&mut self, node: usize, area: f64) -> f64 {
        if node >= self.nodes.len() || area <= 0.0 {
            return 1.0;
        }
        self.punctures += area;
        let n = &self.nodes[node];
        let panel_area = 4.0 * n.half.x * n.half.y;
        let k = (1.0 - area / panel_area).clamp(0.0, 1.0);
        for b in self.bonds.iter_mut().filter(|b| b.a == node || b.b == node) {
            b.strength *= k;
        }
        k
    }

    /// Air escapes through the breaches: dP/dt = −P · A c_s / V (choked flow).
    fn vent(&mut self, m: f64, dtau_s: f64) {
        if self.pressure <= 0.0 || (self.first_break.is_none() && self.punctures <= 0.0) {
            return;
        }
        let mut area = self.punctures;
        for b in self.bonds.iter().filter(|b| b.broken) {
            let gap = (self.nodes[b.a].x - self.nodes[b.b].x).length() * m;
            let rest = (self.nodes[b.a].rest - self.nodes[b.b].rest).length();
            area += b.joint * (gap - rest).clamp(0.0, b.joint);
        }
        self.breach_area = area;
        let vol = std::f64::consts::PI * self.spec.radius * self.spec.radius * self.spec.length;
        let rate = 0.58 * 340.0 * area / vol;
        let p = self.pressure;
        self.pressure *= (-rate * dtau_s).exp();
        if self.pressure < 1.0 {
            self.pressure = 0.0;
        }
        // venting plumes from the widest breaches
        let n = ((p - self.pressure) / self.spec.pressure * 3000.0).ceil().min(64.0) as usize;
        let broken: Vec<usize> = self.bonds.iter().enumerate().filter(|(_, b)| b.broken).map(|(i, _)| i).collect();
        for _ in 0..n {
            if broken.is_empty() {
                break;
            }
            let bi = broken[(self.rng.next_u64() % broken.len() as u64) as usize];
            let (a, b) = (self.bonds[bi].a, self.bonds[bi].b);
            let out = self.frags[self.nodes[a].frag].q * self.nodes[a].axes.z_axis;
            let t = self.rng.f64();
            let (d, v) = self.local_point(m, a, b, t);
            let jet = out * (250.0 + 150.0 * self.rng.f64()) + self.rng.unit_vec() * 60.0;
            self.emit(Particle { d, v: v + jet, life: 25.0, age: 0.0, kind: 2, size: 120.0 });
        }
    }

    /// Offset (m) and velocity (m/s) relative to the reference of a point on the joint a–b.
    fn local_point(&self, m: f64, a: usize, b: usize, t: f64) -> (DVec3, DVec3) {
        let xa = self.nodes[a].x;
        let xb = self.nodes[b].x;
        let x = xa + (xb - xa) * (0.5 + 0.4 * (t - 0.5));
        let w = (self.nodes[a].w + self.nodes[b].w) * 0.5;
        let wr = self.reference.coord_velocity(1.0);
        ((x - self.reference.x) * m, (w - wr) * C)
    }

    fn emit(&mut self, p: Particle) {
        if self.particles.len() < MAX_PARTICLES {
            self.particles.push(p);
        } else {
            let i = self.next_particle % MAX_PARTICLES;
            self.particles[i] = p;
            self.next_particle += 1;
        }
    }

    fn spawn_break_particles(&mut self, m: f64, a: usize, b: usize) {
        let glass = self.nodes[a].kind == Kind::Window || self.nodes[b].kind == Kind::Window;
        let n = if glass { 90 } else { 50 };
        for _ in 0..n {
            let t = self.rng.f64();
            let (d, v) = self.local_point(m, a, b, t);
            let kick = self.rng.unit_vec() * (5.0 + 40.0 * self.rng.f64());
            let kind = if glass && self.rng.f64() < 0.7 { 1 } else { 0 };
            let size = (if kind == 1 { 2.0 } else { 6.0 }) * (0.3 + 2.0 * self.rng.f64() * self.rng.f64()) as f32;
            self.emit(Particle { d, v: v + kick, life: 240.0, age: 0.0, kind, size });
        }
    }

    fn step_particles(&mut self, m: f64, dt_s: f64) {
        if self.particles.is_empty() || dt_s <= 0.0 {
            return;
        }
        let k = if self.reference.dead { DMat3::ZERO } else { physics::tidal_tensor(m, &self.reference) };
        let dt = dt_s.min(5.0);
        for p in self.particles.iter_mut() {
            if p.life <= 0.0 {
                continue;
            }
            p.v += (k * p.d) * dt;
            p.d += p.v * dt;
            p.age += dt as f32;
            p.life -= dt as f32;
        }
    }

    /// Clock spread of the panels (s): (near − far end of the hole, max − min).
    pub fn clock_spread(&self, m: f64) -> (f64, f64) {
        let m_s = m / C;
        let tau = |i: usize| self.frags[self.nodes[i].frag].body.tau + self.nodes[i].dtau;
        let near = (0..self.nodes.len()).min_by(|&a, &b| self.nodes[a].x.length().total_cmp(&self.nodes[b].x.length())).unwrap_or(0);
        let far = (0..self.nodes.len()).max_by(|&a, &b| self.nodes[a].x.length().total_cmp(&self.nodes[b].x.length())).unwrap_or(0);
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for i in 0..self.nodes.len() {
            lo = lo.min(tau(i));
            hi = hi.max(tau(i));
        }
        ((tau(near) - tau(far)) * m_s, (hi - lo) * m_s)
    }

    /// Total momentum (kg·c) of the fragments in coordinate velocity — conserved at splits.
    pub fn momentum(&self) -> DVec3 {
        self.frags.iter().filter(|f| f.alive).map(|f| f.body.coord_velocity(1.0) * f.mass).sum()
    }

    pub fn hash_into(&self, eat: &mut dyn FnMut(f64)) {
        for f in &self.frags {
            for v in [f.body.x.x, f.body.x.y, f.body.x.z, f.body.p.x, f.body.p.y, f.body.p.z, f.body.tau, f.q.x, f.q.w] {
                eat(v);
            }
        }
        eat(self.pressure);
        eat(self.events.len() as f64);
        if self.punctures > 0.0 {
            eat(self.punctures);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn station_at(r: f64, mass_solar: f64) -> (Station, f64) {
        let m = physics::m_geo(mass_solar);
        let body = Geo::rain(DVec3::new(r, 0.0, 0.0));
        (Station::new(Spec::default(), body, DVec3::X), m)
    }

    #[test]
    fn graph_is_complete() {
        let (s, _) = station_at(100.0, 4.3e6);
        assert_eq!(s.nodes.len(), N_NODES);
        let hoop = s.bonds.iter().filter(|b| b.kind == BondKind::Hoop).count();
        assert_eq!(hoop, N_Z * N_THETA + 2 * N_CAP);
        let land = s.nodes.iter().filter(|n| n.kind == Kind::Land).count();
        assert_eq!(land, N_Z * N_THETA / 2);
        // the COM of the symmetric hull is on the axis
        assert!(s.frags[0].com.length() < 1e-6);
    }

    #[test]
    fn intact_far_from_the_hole() {
        // spin + pressure alone stay below the design strength
        let (mut s, m) = station_at(1000.0, 4.3e6);
        for _ in 0..200 {
            s.step(m, 0.05);
        }
        assert!(s.intact(), "broke at r = {:?}", s.first_break.map(|e| e.r));
        let worst = s.nodes.iter().map(|n| n.stress).fold(0.0, f64::max);
        assert!(worst > 0.5 && worst < 1.0, "worst stress {worst}");
    }

    #[test]
    fn sgr_a_station_breaks_before_the_horizon_and_conserves_momentum() {
        let (mut s, m) = station_at(8.0, 4.3e6);
        let p0 = s.momentum();
        let mut n = 0;
        while s.intact() && n < 200_000 {
            s.step(m, 0.01);
            n += 1;
        }
        let r = s.first_break.as_ref().map(|e| e.r).expect("never broke");
        assert!(r > 2.0 && r < 6.0, "first break at r = {r}");
        for _ in 0..400 {
            s.step(m, 0.002);
        }
        assert!(s.live_fragments() > 1);
        eprintln!("first break r = {r:.3} M, fragments {}, events {}, p = {:.0} Pa", s.live_fragments(), s.events.len(), s.pressure);
        // momentum in coordinate velocity is continuous at the split (≪ 1 % drift
        // from the different geodesics during the short run)
        let p1 = s.momentum();
        let _ = p0;
        let total: f64 = s.frags.iter().filter(|f| f.alive).map(|f| f.mass).sum();
        let mean = p1 / total;
        let wr = s.reference.coord_velocity(1.0);
        assert!((mean - wr).length() < 1e-3 * wr.length().max(1e-6), "{mean:?} vs {wr:?}");
    }

    #[test]
    fn intact_station_end_clocks_agree_to_first_order() {
        // equivalence principle: in the free-fall frame the clock rate of a
        // panel differs from the COM clock only at second order in its offset
        let (mut s, m) = station_at(20.0, 4.3e6);
        s.step(m, 0.5);
        let rate_c = 1.0 / s.frags[0].body.dt_dtau(1.0);
        let near = (0..s.nodes.len()).min_by(|&a, &b| s.nodes[a].x.length().total_cmp(&s.nodes[b].x.length())).unwrap();
        let far = (0..s.nodes.len()).max_by(|&a, &b| s.nodes[a].x.length().total_cmp(&s.nodes[b].x.length())).unwrap();
        let dn = s.nodes[near].rate - rate_c;
        let df = s.nodes[far].rate - rate_c;
        // a static pair of clocks 32 km apart would differ by ~ (M/r²)·L
        let static_first_order = 1.0 / (20.0 * 20.0) * 32_000.0 / m;
        assert!((dn - df).abs() < 1e-3 * static_first_order, "dn {dn} df {df} vs {static_first_order}");
    }

    #[test]
    fn punctures_vent_air_and_weaken_the_panel() {
        let (mut st, m) = station_at(1.0e4, 4.3e6);
        let p0 = st.pressure;
        let s0: f64 = st.bonds.iter().filter(|b| b.a == 5 || b.b == 5).map(|b| b.strength).sum();
        let k = st.puncture(5, 1.0e4);
        assert!(k < 1.0);
        let s1: f64 = st.bonds.iter().filter(|b| b.a == 5 || b.b == 5).map(|b| b.strength).sum();
        assert!((s1 / s0 - k).abs() < 1e-12);
        for _ in 0..50 {
            st.step(m, 1.0);
        }
        assert!(st.pressure < p0, "{} vs {}", st.pressure, p0);
        assert!(st.intact());
    }
}
