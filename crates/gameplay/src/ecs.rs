//! A minimal, deterministic, clonable ECS.
//!
//! * `Entity` = index + generation (stale handles are detected).
//! * `Storage<T>` = sparse set: dense arrays of entities and components, a
//!   sparse index table. Iteration is a linear walk over dense memory, the
//!   order depends only on the history of inserts/removes, so it is identical
//!   on every peer.
//! * Everything is `Clone`: a full snapshot of the game for rollback is a
//!   plain `clone()` of the world (a few hundred kB for a match).
//!
//! The world itself (`sim::Ecs`) is a struct with one `Storage` per component
//! type, so systems borrow disjoint storages without runtime borrow checks.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Entity {
    pub index: u32,
    pub generation: u32,
}

impl Entity {
    pub fn bits(self) -> u64 {
        ((self.generation as u64) << 32) | self.index as u64
    }
}

#[derive(Clone, Debug, Default)]
pub struct Entities {
    generation: Vec<u32>,
    alive: Vec<bool>,
    free: Vec<u32>,
    count: usize,
}

impl Entities {
    pub fn spawn(&mut self) -> Entity {
        self.count += 1;
        if let Some(i) = self.free.pop() {
            self.alive[i as usize] = true;
            Entity { index: i, generation: self.generation[i as usize] }
        } else {
            self.generation.push(0);
            self.alive.push(true);
            Entity { index: (self.generation.len() - 1) as u32, generation: 0 }
        }
    }
    pub fn despawn(&mut self, e: Entity) -> bool {
        if !self.is_alive(e) {
            return false;
        }
        let i = e.index as usize;
        self.alive[i] = false;
        self.generation[i] = self.generation[i].wrapping_add(1);
        // FIFO-ish reuse keeps handles fresh longer; order is deterministic either way
        self.free.insert(0, e.index);
        self.count -= 1;
        true
    }
    pub fn is_alive(&self, e: Entity) -> bool {
        let i = e.index as usize;
        i < self.alive.len() && self.alive[i] && self.generation[i] == e.generation
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

const NONE: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct Storage<T> {
    sparse: Vec<u32>,
    entities: Vec<Entity>,
    data: Vec<T>,
}

impl<T> Default for Storage<T> {
    fn default() -> Self {
        Storage { sparse: Vec::new(), entities: Vec::new(), data: Vec::new() }
    }
}

impl<T> Storage<T> {
    pub fn insert(&mut self, e: Entity, v: T) {
        let i = e.index as usize;
        if i >= self.sparse.len() {
            self.sparse.resize(i + 1, NONE);
        }
        let d = self.sparse[i];
        if d != NONE {
            self.entities[d as usize] = e;
            self.data[d as usize] = v;
        } else {
            self.sparse[i] = self.data.len() as u32;
            self.entities.push(e);
            self.data.push(v);
        }
    }
    pub fn remove(&mut self, e: Entity) -> Option<T> {
        let i = e.index as usize;
        let d = *self.sparse.get(i)?;
        if d == NONE || self.entities[d as usize] != e {
            return None;
        }
        let last = self.data.len() - 1;
        self.entities.swap(d as usize, last);
        self.data.swap(d as usize, last);
        let moved = self.entities[d as usize];
        self.sparse[moved.index as usize] = d;
        self.sparse[i] = NONE;
        self.entities.pop();
        self.data.pop()
    }
    #[inline]
    fn slot(&self, e: Entity) -> Option<usize> {
        let d = *self.sparse.get(e.index as usize)?;
        (d != NONE && self.entities[d as usize] == e).then_some(d as usize)
    }
    pub fn get(&self, e: Entity) -> Option<&T> {
        self.slot(e).map(|d| &self.data[d])
    }
    pub fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
        self.slot(e).map(|d| &mut self.data[d])
    }
    pub fn contains(&self, e: Entity) -> bool {
        self.slot(e).is_some()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.entities.iter().copied().zip(self.data.iter())
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Entity, &mut T)> {
        self.entities.iter().copied().zip(self.data.iter_mut())
    }
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }
    pub fn values(&self) -> &[T] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_set_basics() {
        let mut es = Entities::default();
        let mut s: Storage<i32> = Storage::default();
        let a = es.spawn();
        let b = es.spawn();
        let c = es.spawn();
        s.insert(a, 1);
        s.insert(b, 2);
        s.insert(c, 3);
        assert_eq!(s.remove(a), Some(1));
        assert_eq!(s.get(c), Some(&3));
        assert!(es.despawn(a));
        let a2 = es.spawn();
        assert_eq!(a2.index, a.index);
        assert_ne!(a2, a);
        assert!(s.get(a2).is_none());
        let v: Vec<i32> = s.iter().map(|(_, v)| *v).collect();
        assert_eq!(v, vec![3, 2]);
    }
}
