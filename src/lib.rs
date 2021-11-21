mod components;
mod systems;

#[cfg(test)]
mod tests {
    use legion::prelude::*;
    use nalgebra as na;

    #[test]
    fn test_na() {
        let velocity = na::RowVector2::new(5.0_f32, 0.0);
        let rot = na::Rotation2::new(std::f32::consts::FRAC_PI_2);
        let rotated_velocity = velocity * rot;
        approx::assert_relative_eq!(*rotated_velocity.get((0, 0)).unwrap(), 0.0, epsilon = 0.001);
        approx::assert_relative_eq!(
            *rotated_velocity.get((0, 1)).unwrap(),
            -5.0,
            epsilon = 0.001
        );
    }

    #[test]
    fn test_movement_system() {
        let universe = Universe::new();
        let mut world = universe.create_world();
        world.insert(
            (),
            vec![(
                super::components::Transformation {
                    location: na::Vector2::new(5.0, 5.0),
                    scale: na::Vector2::new(1.0, 1.0),
                    rotation: 0.0,
                },
                super::components::Velocity { velocity: 5.0 },
            )],
        );
        let mut res = Resources::default();
        let sys = super::systems::build_movement_system();
        let mut schedule = Schedule::builder().add_system(sys).flush().build();
        schedule.execute(&mut world, &mut res);
        for one_entity in world.iter_entities().take(1) {
            assert!(world
                .get_component::<super::components::Transformation>(one_entity)
                .is_some());
            approx::assert_relative_eq!(
                world
                    .get_component::<super::components::Transformation>(one_entity)
                    .unwrap()
                    .location,
                na::Vector2::new(10.0, 5.0)
            );
            assert!(world
                .get_component::<super::components::Velocity>(one_entity)
                .is_some());
        }
    }

    #[test]
    fn test_collision_system() {
        let universe = Universe::new();
        let mut resources = Resources::default();
        resources.insert(ncollide2d::pipeline::CollisionWorld::<f32, ()>::new(1.0));
        resources.insert(std::collections::HashMap::<
            ncollide2d::pipeline::CollisionObjectSlabHandle,
            Entity,
        >::new());
        resources.insert(Vec::<[Entity; 2]>::new());
        let mut world = universe.create_world();
        world.insert(
            (),
            vec![
                (
                    super::components::Transformation {
                        location: na::Vector2::new(0.0, 0.0),
                        rotation: 0.0,
                        scale: na::Vector2::new(1.0, 1.0),
                    },
                    super::components::Hitbox {
                        slap_handle: None,
                        shape: std::sync::Arc::new(ncollide2d::shape::Cuboid::new(
                            na::Vector2::new(5.0_f32, 5.0),
                        )),
                    },
                ),
                (
                    super::components::Transformation {
                        location: na::Vector2::new(1.0, 2.0),
                        rotation: 0.0,
                        scale: na::Vector2::new(1.0, 1.0),
                    },
                    super::components::Hitbox {
                        slap_handle: None,
                        shape: std::sync::Arc::new(ncollide2d::shape::Cuboid::new(
                            na::Vector2::new(5.0, 5.0),
                        )),
                    },
                ),
            ],
        );
        let mut schedule = Schedule::builder()
            .add_system(super::systems::build_map_entity_collision_handle_system())
            .add_system(super::systems::build_collision_system())
            .flush()
            .build();
        schedule.execute(&mut world, &mut resources);
        let resources_len = resources.get::<Vec<[Entity; 2]>>().unwrap().len();
        assert_eq!(resources_len, 1);
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct TestTag;
    struct TestComponent;
    #[test]
    fn test_ball_paddle_collision() {
        let universe = Universe::new();
        let mut world = universe.create_world();
        let ball: Vec<_> = world
            .insert(
                (super::components::Ball, TestTag),
                vec![(
                    super::components::Transformation {
                        location: na::Vector2::new(0.0, 0.0),
                        rotation: 0.0,
                        scale: na::Vector2::new(1.0, 1.0),
                    },
                    TestComponent,
                )],
            )
            .to_vec();
        let paddle = world.insert(
            (super::components::Barrier, TestTag),
            vec![(
                super::components::Transformation {
                    location: na::Vector2::new(0.0, 0.0),
                    rotation: 0.0,
                    scale: na::Vector2::new(1.0, 1.0),
                },
                TestComponent,
            )],
        );
        let mut resources = Resources::default();
        let entities = [ball[0], paddle[0]];
        resources.insert(vec![entities]);
        let mut schedule = Schedule::builder()
            .add_system(super::systems::build_handle_ball_barrier_collision())
            .flush()
            .build();
        schedule.execute(&mut world, &mut resources);
        let trans = world
            .get_component::<super::components::Transformation>(ball[0])
            .unwrap();
        approx::assert_relative_eq!(trans.rotation, std::f32::consts::PI);
    }
}
