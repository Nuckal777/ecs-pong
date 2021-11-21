#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::type_complexity)]

use glutin::dpi;
use glutin::event::Event;
use glutin::event::WindowEvent;
use glutin::event_loop::ControlFlow;
use glutin::event_loop::EventLoop;
use glutin::window;
use graphics::gl;
use graphics::Program;
use legion::prelude::*;
use nalgebra as na;
use std::ffi::CStr;

mod components;
mod graphics;
mod systems;

fn main() {
    let el = EventLoop::new();
    let wb = window::WindowBuilder::new()
        .with_title("Hello world!")
        .with_inner_size(dpi::PhysicalSize::new(1024, 768));
    let windowed_context = glutin::ContextBuilder::new()
        .build_windowed(wb, &el)
        .unwrap();
    let current_context = unsafe {
        windowed_context
            .make_current()
            .expect("Cannot make OpenGL context current")
    };
    let gl = gl::Gl::load_with(|s| current_context.context().get_proc_address(s));
    let shader = init_shader(&gl);

    let mut vertecies = graphics::VertexArray::with_vertecies(
        gl.clone(),
        &[
            graphics::Vertex {
                color: [1.0, 1.0, 1.0],
                vertex: [250.0, 250.0],
            },
            graphics::Vertex {
                color: [1.0, 1.0, 1.0],
                vertex: [350.0, 250.0],
            },
            graphics::Vertex {
                color: [1.0, 1.0, 1.0],
                vertex: [350.0, 350.0],
            },
        ],
    );

    let universe = Universe::new();
    let mut world = universe.create_world();
    let mut resources = Resources::default();
    resources.insert(Vec::<components::RenderInfo>::new());
    resources.insert(ncollide2d::pipeline::CollisionWorld::<f32, ()>::new(1.0));
    resources.insert(std::collections::HashMap::<
        ncollide2d::pipeline::CollisionObjectSlabHandle,
        Entity,
    >::new());
    resources.insert(Vec::<[Entity; 2]>::new());
    let mut schedule = Schedule::builder()
        .add_system(systems::build_map_entity_collision_handle_system())
        .add_system(systems::build_movement_system())
        .add_system(systems::build_collision_system())
        .add_system(systems::build_handle_ball_barrier_collision())
        .add_system(systems::build_dispatch_render_system())
        .flush()
        .build();

    insert_components(&mut world);

    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        schedule.execute(&mut world, &mut resources);
        let mut vertex_data = Vec::<graphics::Vertex>::new();
        for one_info in resources
            .get::<Vec<components::RenderInfo>>()
            .unwrap()
            .iter()
        {
            for one_vertex in one_info.shape.as_slice().iter().take(3) {
                vertex_data.push(graphics::Vertex {
                    color: one_info.color,
                    vertex: (*one_vertex).into(),
                });
            }
        }
        for one_info in resources
            .get::<Vec<components::RenderInfo>>()
            .unwrap()
            .iter()
        {
            for one_vertex in one_info.shape.as_slice().iter().skip(1).take(3) {
                vertex_data.push(graphics::Vertex {
                    color: one_info.color,
                    vertex: (*one_vertex).into(),
                });
            }
        }
        vertecies.store(&vertex_data);

        match event {
            Event::LoopDestroyed => return,
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(physical_size) => current_context.resize(physical_size),
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                _ => (),
            },
            _ => (),
        }

        unsafe {
            gl.Clear(gl::COLOR_BUFFER_BIT);
            gl.ClearColor(0.0, 0.0, 1.0, 0.5);
            shader.bind();
            vertecies.draw(vertex_data.len() as i32);
        }
        current_context.swap_buffers().unwrap();
    });
}

fn init_shader(gl: &gl::Gl) -> Program {
    let version = unsafe {
        let data = CStr::from_ptr(gl.GetString(gl::VERSION).cast())
            .to_bytes()
            .to_vec();
        String::from_utf8(data).unwrap()
    };
    println!("OpenGL version {}", version);
    let mut shader = graphics::Program::new(gl.clone());
    shader.model = na::Matrix4::identity();
    shader.view = na::Matrix4::identity();
    shader.proj = *na::Orthographic3::new(0.0, 1024.0, 768.0, 0.0, 1.0, 20.0).as_matrix();
    shader
}

fn insert_components(world: &mut World) {
    world.insert(
        (components::Ball, ()),
        vec![(
            components::Transformation {
                location: na::Vector2::new(300.0, 300.0),
                rotation: 0.0,
                scale: na::Vector2::new(1.0, 1.0),
            },
            components::RenderShape {
                color: [1.0, 0.0, 0.0],
                half_extents: na::Vector2::new(10.0, 10.0),
            },
            components::Velocity { velocity: 0.03 },
            components::Hitbox {
                shape: std::sync::Arc::new(ncollide2d::shape::Cuboid::new(na::Vector2::new(
                    10.0, 10.0,
                ))),
                slap_handle: None,
            },
        )],
    );
    world.insert(
        (components::Barrier, ()),
        vec![
            (
                components::Transformation {
                    location: na::Vector2::new(600.0, 300.0),
                    rotation: 0.0,
                    scale: na::Vector2::new(1.0, 1.0),
                },
                components::RenderShape {
                    color: [0.0, 1.0, 1.0],
                    half_extents: na::Vector2::new(2.0, 40.0),
                },
                components::Hitbox {
                    shape: std::sync::Arc::new(ncollide2d::shape::Cuboid::new(na::Vector2::new(
                        2.0, 40.0,
                    ))),
                    slap_handle: None,
                },
            ),
            (
                components::Transformation {
                    location: na::Vector2::new(50.0, 300.0),
                    rotation: 0.0,
                    scale: na::Vector2::new(1.0, 1.0),
                },
                components::RenderShape {
                    color: [0.0, 1.0, 1.0],
                    half_extents: na::Vector2::new(2.0, 40.0),
                },
                components::Hitbox {
                    shape: std::sync::Arc::new(ncollide2d::shape::Cuboid::new(na::Vector2::new(
                        2.0, 40.0,
                    ))),
                    slap_handle: None,
                },
            ),
        ],
    );
}
