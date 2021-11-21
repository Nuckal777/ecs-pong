use nalgebra as na;

// Actual components
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transformation {
    pub location: na::Vector2<f32>,
    pub scale: na::Vector2<f32>,
    pub rotation: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Velocity {
    pub velocity: f32,
}

pub struct Hitbox {
    pub shape: std::sync::Arc<dyn ncollide2d::shape::Shape<f32>>,
    pub slap_handle: Option<ncollide2d::pipeline::CollisionObjectSlabHandle>,
}

pub struct RenderInfo {
    pub shape: Vec<na::Vector2<f32>>,
    pub color: [f32; 3],
}

pub struct RenderShape {
    pub color: [f32; 3],
    pub half_extents: na::Vector2<f32>,
}

// Tags
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Barrier;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Goal;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball;
