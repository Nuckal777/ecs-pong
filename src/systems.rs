use super::components as comps;
use legion::filter;
use legion::prelude::*;
use nalgebra as na;
use ncollide2d::world;

pub fn build_movement_system() -> Box<dyn Schedulable> {
    SystemBuilder::<()>::new("movement")
        .write_component::<comps::Transformation>()
        .read_component::<comps::Velocity>()
        .with_query(<(Read<comps::Velocity>, Write<comps::Transformation>)>::query())
        .build(do_movement)
}

fn do_movement(
    _: &mut CommandBuffer,
    world: &mut SubWorld,
    _: &mut <() as legion::systems::resource::ResourceSet>::PreparedResources,
    queries: &mut Query<
        (Read<comps::Velocity>, Write<comps::Transformation>),
        filter::EntityFilterTuple<
            filter::And<(
                filter::ComponentFilter<comps::Velocity>,
                filter::ComponentFilter<comps::Transformation>,
            )>,
            filter::And<(filter::Passthrough, filter::Passthrough)>,
            filter::And<(filter::Passthrough, filter::Passthrough)>,
        >,
    >,
) {
    for mut comp in queries.iter_mut(&mut *world) {
        let target = comp.1.location
            + na::Vector2::new(
                comp.1.rotation.cos() * comp.0.velocity,
                comp.1.rotation.sin() * comp.0.velocity,
            );
        comp.1.location = target;
    }
}

pub fn build_map_entity_collision_handle_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("map_entity_collision_handle")
        .write_component::<comps::Hitbox>()
        .write_resource::<std::collections::HashMap<
            ncollide2d::pipeline::CollisionObjectSlabHandle,
            legion::entity::Entity,
        >>()
        .write_resource::<world::CollisionWorld<f32, ()>>()
        .with_query(<Write<comps::Hitbox>>::query())
        .build(map_entity_collision_handle)
}

pub fn map_entity_collision_handle(
    _: &mut CommandBuffer,
    world: &mut SubWorld,
    resource: &mut (
        legion::systems::resource::PreparedWrite<
            std::collections::HashMap<
                ncollide2d::pipeline::CollisionObjectSlabHandle,
                legion::entity::Entity,
            >,
        >,
        legion::systems::resource::PreparedWrite<
            ncollide2d::pipeline::world::CollisionWorld<f32, ()>,
        >,
    ),
    query: &mut Query<
        Write<comps::Hitbox>,
        filter::EntityFilterTuple<
            filter::ComponentFilter<comps::Hitbox>,
            filter::Passthrough,
            filter::Passthrough,
        >,
    >,
) {
    for (entity, mut hitbox) in query.iter_entities_mut(world) {
        if hitbox.slap_handle.is_some() {
            continue;
        }
        let map = &mut resource.0;
        let world = &mut resource.1;
        let shape_handle = ncollide2d::shape::ShapeHandle::<f32>::from_arc(hitbox.shape.clone());
        let (handle, co) = world.add(
            na::Isometry2::new(na::Vector2::new(0.0, 0.0), 0.0),
            shape_handle,
            ncollide2d::pipeline::CollisionGroups::new(),
            ncollide2d::pipeline::GeometricQueryType::Proximity(1.0),
            (),
        );
        hitbox.slap_handle = Some(handle);
        let old = map.insert(handle, entity);
        if old.is_some() {
            panic!("entries in the entity collision handle map should logically not be swapped");
        }
    }
}

pub fn build_collision_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("collision")
        .read_component::<comps::Transformation>()
        .write_component::<comps::Hitbox>()
        .read_resource::<std::collections::HashMap<
            ncollide2d::pipeline::CollisionObjectSlabHandle,
            legion::entity::Entity,
        >>()
        .write_resource::<world::CollisionWorld<f32, ()>>()
        .write_resource::<std::vec::Vec<[Entity; 2]>>()
        .with_query(<(Read<comps::Transformation>, Write<comps::Hitbox>)>::query())
        .build(do_collision)
}

pub fn do_collision(
    _: &mut CommandBuffer,
    world: &mut SubWorld,
    resource: &mut (
        legion::systems::resource::PreparedRead<
            std::collections::HashMap<
                ncollide2d::pipeline::CollisionObjectSlabHandle,
                legion::entity::Entity,
            >,
        >,
        legion::systems::resource::PreparedWrite<
            ncollide2d::pipeline::world::CollisionWorld<f32, ()>,
        >,
        legion::systems::resource::PreparedWrite<std::vec::Vec<[Entity; 2]>>,
    ),
    query: &mut Query<
        (Read<comps::Transformation>, Write<comps::Hitbox>),
        filter::EntityFilterTuple<
            filter::And<(
                filter::ComponentFilter<comps::Transformation>,
                filter::ComponentFilter<comps::Hitbox>,
            )>,
            filter::And<(filter::Passthrough, filter::Passthrough)>,
            filter::And<(filter::Passthrough, filter::Passthrough)>,
        >,
    >,
) {
    let (map, co_world, colliders) = resource;
    // update shape position
    for comp in query.iter_mut(world) {
        let trans: legion::borrow::Ref<comps::Transformation> = comp.0;
        let mut hb: legion::borrow::RefMut<comps::Hitbox> = comp.1;
        if let Some(handle) = hb.slap_handle {
            if let Some(co) = co_world.objects.get_mut(handle) {
                co.set_position(na::Isometry2::new(trans.location, trans.rotation));
            }
        }
    }
    // do collision test
    co_world.update();
    // record collisions
    colliders.clear();
    for collision in co_world.proximity_pairs(true) {
        let entities = [map[&collision.0], map[&collision.1]];
        colliders.push(entities);
    }
}

fn sort_collision_pair_by_tag<T1, T2>(
    world: &mut SubWorld,
    pair: &[Entity; 2],
) -> Option<(Entity, Entity)>
where
    T1: Clone + PartialEq + Send + Sync + 'static,
    T2: Clone + PartialEq + Send + Sync + 'static,
{
    let opt_first = pair
        .iter()
        .enumerate()
        .find(|e| world.get_tag::<T1>(*e.1).is_some());
    if let Some(first) = opt_first {
        let second_idx = 1 - first.0;
        if world.get_tag::<T2>(pair[second_idx]).is_some() {
            return Some((*first.1, pair[second_idx]));
        }
    }
    None
}

pub fn build_handle_ball_barrier_collision() -> Box<dyn Schedulable> {
    SystemBuilder::new("handle_ball_barrier_collision")
        .read_resource::<std::vec::Vec<[Entity; 2]>>()
        .write_component::<comps::Transformation>()
        .build(handle_ball_barrier_collision)
}

fn handle_ball_barrier_collision(
    _: &mut CommandBuffer,
    world: &mut SubWorld,
    colliders: &mut legion::systems::resource::PreparedRead<std::vec::Vec<[Entity; 2]>>,
    _: &mut (),
) {
    for one_collision in colliders.iter() {
        if let Some((ball, _)) =
            sort_collision_pair_by_tag::<comps::Ball, comps::Barrier>(world, one_collision)
        {
            world
                .get_component_mut::<comps::Transformation>(ball)
                .unwrap()
                .rotation += std::f32::consts::PI;
        }
    }
}

pub fn build_dispatch_render_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("dispatch_render")
        .read_component::<comps::Transformation>()
        .read_component::<comps::RenderShape>()
        .write_resource::<Vec<comps::RenderInfo>>()
        .with_query(<(
            Read<comps::Transformation>,
            Read<comps::RenderShape>,
        )>::query())
        .build(dispatch_render)
}

pub fn dispatch_render(
    _: &mut CommandBuffer,
    world: &mut SubWorld,
    infos: &mut legion::systems::resource::PreparedWrite<Vec<comps::RenderInfo>>,
    query: &mut Query<
        (
            Read<comps::Transformation>,
            Read<comps::RenderShape>,
        ),
        filter::EntityFilterTuple<
            filter::And<(
                filter::ComponentFilter<comps::Transformation>,
                filter::ComponentFilter<comps::RenderShape>,
            )>,
            filter::And<(
                filter::Passthrough,
                filter::Passthrough,
            )>,
            filter::And<(
                filter::Passthrough,
                filter::Passthrough,
            )>,
        >,
    >,
) {
    infos.clear();
    for one_entity in query.iter(world) {
        let trans: legion::borrow::Ref<comps::Transformation> = one_entity.0;
        let shape: legion::borrow::Ref<comps::RenderShape> = one_entity.1;
        infos.push(comps::RenderInfo {
            shape: vec![
                trans.location + na::Vector2::new(-shape.half_extents[0], shape.half_extents[1]),
                trans.location - shape.half_extents,
                trans.location + shape.half_extents,
                trans.location + na::Vector2::new(shape.half_extents[0], -shape.half_extents[1]),
            ],
            color: shape.color,
        });
    }
}
