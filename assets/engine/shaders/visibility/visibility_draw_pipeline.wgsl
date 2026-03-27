#if MESHLET_DEPTH_PASS
@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> visible_meshlets: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(4) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(5) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(6) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(7) var<uniform> visibility_bucket_info: VisibilityBucketInfo;
#endif

#if MESHLET_RASTER_PASS
@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> visible_meshlets: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(4) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(5) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(6) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(7) var<uniform> visibility_bucket_info: VisibilityBucketInfo;
#endif

#if MESHLET_RESOLVE_PASS
@group(1) @binding(0) var visibility_entity_texture: texture_2d<u32>;
@group(1) @binding(1) var visibility_surface_texture: texture_2d<u32>;
@group(1) @binding(2) var visibility_bucket_texture: texture_2d<u32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(6) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(7) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(8) var<uniform> visibility_bucket_info: VisibilityBucketInfo;
#endif

#ifndef CUSTOM_DEPTH_VS
fn depth_vertex(v_out: ptr<function, DepthVertexOutput>) -> DepthVertexOutput {
    return *v_out;
}
#endif

#ifndef CUSTOM_RASTER_VS
fn raster_vertex(v_out: ptr<function, RasterVertexOutput>) -> RasterVertexOutput {
    return *v_out;
}
#endif

#ifndef CUSTOM_DEPTH_FRAGMENT_MASK
fn depth_fragment_mask(input: DepthVertexOutput) -> f32 {
    return 1.0;
}
#endif

#ifndef CUSTOM_RASTER_FRAGMENT_MASK
fn raster_fragment_mask(input: RasterVertexOutput) -> f32 {
    return 1.0;
}
#endif

#ifndef CUSTOM_RASTER_FRAGMENT
fn raster_fragment(v_out: RasterVertexOutput, f_out: ptr<function, RasterFragmentOutput>) -> RasterFragmentOutput {
    return *f_out;
}
#endif

#ifndef CUSTOM_RESOLVE_FRAGMENT
fn resolve_fragment(
    v_out: ResolveFragmentInput,
    f_out: ptr<function, ResolveFragmentOutput>
) -> ResolveFragmentOutput {
    return *f_out;
}
#endif

#if MESHLET_DEPTH_PASS
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DepthVertexOutput {
    let visible_entry = visible_meshlets[ii];
    let global_meshlet_index = meshlet_index(visible_entry);
    let meshlet = meshlets[global_meshlet_index];

    let triangle_index = vi / 3u;
    let corner_index = vi % 3u;

    var output: DepthVertexOutput;
    output.position = vec4<f32>(-2.0, -2.0, 1.0, 1.0);
    output.uv = vec2<f32>(0.0);
    output.entity_id = INVALID_IDX;
    output.section_index = 0u;

    if (triangle_index >= meshlet.triangle_count) {
        return output;
    }

    let object_instance_index = meshlet_object_index(visible_entry);
    if (object_instances[object_instance_index].visibility_bucket != visibility_bucket_info.current_visibility_bucket) {
        return output;
    }

    let local_triangle_index = meshlet.triangle_offset + triangle_index * 3u + corner_index;
    let local_vertex_index = meshlet_triangles[local_triangle_index];
    let global_vertex_index = meshlet_vertices[meshlet.vertex_offset + local_vertex_index];

    let entity_row = get_entity_row(object_instances[object_instance_index].row);
    let entity_resolved = entity_index_lookup[entity_row];
    let transform = entity_transforms[entity_resolved].transform;
    let world_position = transform * vertex_position4(vertex_buffer[global_vertex_index]);

    let section_index = u32(max(vertex_section_index(vertex_buffer[global_vertex_index]), 0.0));

    output.position = view_buffer[u32(frame_info.view_index)].view_projection_matrix * world_position;
    output.uv = vertex_uv(vertex_buffer[global_vertex_index]);
    output.entity_id = entity_resolved;
    output.section_index = section_index;

    output = depth_vertex(&output);

    return output;
}
#endif

#if MESHLET_RASTER_PASS
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RasterVertexOutput {
    let visible_entry = visible_meshlets[ii];
    let global_meshlet_index = meshlet_index(visible_entry);
    let meshlet = meshlets[global_meshlet_index];

    let triangle_index = vi / 3u;
    let corner_index = vi % 3u;

    var output: RasterVertexOutput;
    output.position = vec4<f32>(-2.0, -2.0, 1.0, 1.0);
    output.uv = vec2<f32>(0.0);
    output.entity_id = INVALID_IDX;
    output.section_index = 0u;

    output.barycentric = vec2<f32>(0.0, 0.0);
    output.meshlet_index = global_meshlet_index;
    output.triangle_index = triangle_index;

    if (triangle_index >= meshlet.triangle_count) {
        return output;
    }

    let object_instance_index = meshlet_object_index(visible_entry);
    if (object_instances[object_instance_index].visibility_bucket != visibility_bucket_info.current_visibility_bucket) {
        return output;
    }

    let local_triangle_index = meshlet.triangle_offset + triangle_index * 3u + corner_index;
    let local_vertex_index = meshlet_triangles[local_triangle_index];
    let global_vertex_index = meshlet_vertices[meshlet.vertex_offset + local_vertex_index];

    let entity_row = get_entity_row(object_instances[object_instance_index].row);
    let entity_resolved = entity_index_lookup[entity_row];
    let transform = entity_transforms[entity_resolved].transform;
    let world_position = transform * vertex_position4(vertex_buffer[global_vertex_index]);

    let section_index = u32(max(vertex_section_index(vertex_buffer[global_vertex_index]), 0.0));

    output.position = view_buffer[u32(frame_info.view_index)].view_projection_matrix * world_position;
    output.uv = vertex_uv(vertex_buffer[global_vertex_index]);
    output.entity_id = entity_resolved;
    output.section_index = section_index;

    output.barycentric = select(
        select(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), corner_index == 1u),
        vec2<f32>(1.0, 0.0),
        corner_index == 0u
    );

    output = raster_vertex(&output);

    return output;
}
#endif

#if MESHLET_RESOLVE_PASS
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> ResolveVertexOutput {
    var output: ResolveVertexOutput;
    output.position = vertex_position4(vertex_buffer[vi]);
    output.uv = vertex_uv(vertex_buffer[vi]);
    return output;
}
#endif

#if MESHLET_DEPTH_PASS
@fragment
fn fs(input: DepthVertexOutput) {
    let mask = depth_fragment_mask(input);
    if (mask <= 0.0) {
        discard;
    }
}
#endif

#if MESHLET_RASTER_PASS
@fragment
fn fs(input: RasterVertexOutput) -> RasterFragmentOutput {
    let mask = raster_fragment_mask(input);
    if (mask <= 0.0) {
        discard;
    }

    var output = RasterFragmentOutput(
        input.entity_id,
        pack_surface_id(input.meshlet_index, input.triangle_index),
        visibility_bucket_info.current_visibility_bucket
    );

    return raster_fragment(input, &output);
}
#endif

#if MESHLET_RESOLVE_PASS
@fragment
fn fs(input: ResolveVertexOutput) -> ResolveFragmentOutput {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    let pixel_coord = uv_to_coord(input.uv, resolution);

    let entity_id = textureLoad(visibility_entity_texture, pixel_coord, 0).x;
    if (entity_id == INVALID_IDX) {
        discard;
    }

    let visibility_bucket = textureLoad(visibility_bucket_texture, pixel_coord, 0).x;
    if (visibility_bucket != visibility_bucket_info.current_visibility_bucket) {
        discard;
    }

    let surface = textureLoad(visibility_surface_texture, pixel_coord, 0).x;
    let meshlet_index_value = unpack_surface_meshlet(surface);
    let triangle_index = unpack_surface_triangle(surface);
    let meshlet = meshlets[meshlet_index_value];

    let tri_offset = meshlet.triangle_offset + triangle_index * 3u;
    let local_index0 = meshlet_triangles[tri_offset + 0u];
    let local_index1 = meshlet_triangles[tri_offset + 1u];
    let local_index2 = meshlet_triangles[tri_offset + 2u];

    let global_index0 = meshlet_vertices[meshlet.vertex_offset + local_index0];
    let global_index1 = meshlet_vertices[meshlet.vertex_offset + local_index1];
    let global_index2 = meshlet_vertices[meshlet.vertex_offset + local_index2];

    let decoded0 = decode_vertex(vertex_buffer[global_index0]);
    let decoded1 = decode_vertex(vertex_buffer[global_index1]);
    let decoded2 = decode_vertex(vertex_buffer[global_index2]);

    let section_index = u32(max(decoded0.section_index, 0.0));
    let entity_transform = entity_transforms[entity_id];
    let view_index = u32(frame_info.view_index);

    let world_position0 = entity_transform.transform * decoded0.position;
    let world_position1 = entity_transform.transform * decoded1.position;
    let world_position2 = entity_transform.transform * decoded2.position;

    let current_clip_pos0 = view_buffer[view_index].view_projection_matrix * world_position0;
    let current_clip_pos1 = view_buffer[view_index].view_projection_matrix * world_position1;
    let current_clip_pos2 = view_buffer[view_index].view_projection_matrix * world_position2;


    let bary = calc_full_barycentric(
        coord_to_uv(pixel_coord, resolution),
        current_clip_pos0,
        current_clip_pos1,
        current_clip_pos2
    );

    let local_position = interpolate_vec3(
        decoded0.position.xyz,
        decoded1.position.xyz,
        decoded2.position.xyz,
        bary
    );
    let prev_local_position = local_position;
    let uv = interpolate_vec2(decoded0.uv, decoded1.uv, decoded2.uv, bary);
    let normal_os = safe_normalize(interpolate_vec3(
        decoded0.normal.xyz,
        decoded1.normal.xyz,
        decoded2.normal.xyz,
        bary
    ));
    let tangent_os = safe_normalize(interpolate_vec3(
        decoded0.tangent.xyz,
        decoded1.tangent.xyz,
        decoded2.tangent.xyz,
        bary
    ));
    let bitangent_os = safe_normalize(interpolate_vec3(
        decoded0.bitangent.xyz,
        decoded1.bitangent.xyz,
        decoded2.bitangent.xyz,
        bary
    ));

    let world_position = vec4<f32>(interpolate_vec3(
        world_position0.xyz,
        world_position1.xyz,
        world_position2.xyz,
        bary
    ), 1.0);
    let prev_world_position = entity_transform.prev_transform * vec4<f32>(prev_local_position, 1.0);

    let view_proj = view_buffer[view_index].view_projection_matrix;
    let prev_view_proj = view_buffer[view_index].prev_projection_matrix * view_buffer[view_index].prev_view_matrix;
    let current_clip_pos = view_proj * world_position;
    let prev_clip_pos = prev_view_proj * prev_world_position;

    let normal_ws = safe_normalize((entity_transform.transform * vec4<f32>(normal_os, 0.0)).xyz);
    let tangent_ws = safe_normalize((entity_transform.transform * vec4<f32>(tangent_os, 0.0)).xyz);
    let bitangent_ws = safe_normalize((entity_transform.transform * vec4<f32>(bitangent_os, 0.0)).xyz);

    var material_input: ResolveFragmentInput;
    material_input.screen_uv = input.uv;
    material_input.device_depth = textureLoad(depth_texture, pixel_coord, 0).x;
    material_input.entity_id = entity_id;
    material_input.section_index = section_index;
    material_input.meshlet_index = meshlet_index_value;
    material_input.triangle_index = triangle_index;
    material_input.barycentric = bary;
    material_input.local_position = vec4<f32>(local_position, 1.0);
    material_input.prev_local_position = vec4<f32>(prev_local_position, 1.0);
    material_input.world_position = world_position;
    material_input.prev_world_position = prev_world_position;
    material_input.uv = uv;
    material_input.normal = vec4<f32>(normal_ws, 0.0);
    material_input.tangent = vec4<f32>(tangent_ws, 0.0);
    material_input.bitangent = vec4<f32>(bitangent_ws, 0.0);
    material_input.current_clip_pos = current_clip_pos;
    material_input.prev_clip_pos = prev_clip_pos;

    let current_ndc = current_clip_pos.xy / current_clip_pos.w;
    let prev_ndc = prev_clip_pos.xy / prev_clip_pos.w;
    let motion = current_ndc - prev_ndc;

    var output: ResolveFragmentOutput;
    output.albedo = vec4<f32>(0.0);
    output.smra = vec4<f32>(0.0);
    output.normal = vec4<f32>(material_input.normal.xyz, 1.0);
    output.motion_emissive = vec4<f32>(motion, 0.0, 0.0);

    return resolve_fragment(material_input, &output);
}
#endif
