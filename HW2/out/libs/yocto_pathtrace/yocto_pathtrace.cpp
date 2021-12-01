//
// Implementation for Yocto/PathTrace.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "yocto_pathtrace.h"

#include <yocto/yocto_cli.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_math.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_shading.h>
#include <yocto/yocto_shape.h>

#if YOCTO_DENOISE
#include <yocto/OpenImageDenoise/oidn.hpp>
#endif

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace yocto {

// Convenience functions
[[maybe_unused]] static vec3f eval_position(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_normal(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_element_normal(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_element_normal(
      scene, scene.instances[intersection.instance], intersection.element);
}
[[maybe_unused]] static vec3f eval_shading_position(const scene_data& scene,
    const bvh_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec3f eval_shading_normal(const scene_data& scene,
    const bvh_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec2f eval_texcoord(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_texcoord(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static material_point eval_material(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_material(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static bool is_volumetric(
    const scene_data& scene, const bvh_intersection& intersection) {
  return is_volumetric(scene, scene.instances[intersection.instance]);
}

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_emission(const material_point& material, const vec3f& normal,
    const vec3f& outgoing) {
  return dot(normal, outgoing) >= 0 ? material.emission : vec3f{0, 0, 0};
}

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  // YOUR CODE GOES HERE
  if (material.roughness == 0) return {0, 0, 0};

  if (material.type == material_type::matte) {
    return eval_matte(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::glossy) {
    return eval_glossy(material.color, material.ior, material.roughness, normal,
        outgoing, incoming);
  } else if (material.type == material_type::reflective) {
    return eval_reflective(
        material.color, material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return eval_transparent(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return eval_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else {
    return {0, 0, 0};
  }
}

static vec3f eval_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  // YOUR CODE GOES HERE
  if (material.roughness != 0) return {0, 0, 0};

  if (material.type == material_type::reflective) {
    return eval_reflective(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return eval_transparent(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return eval_refractive(
        material.color, material.ior, normal, outgoing, incoming);
  } else {
    return {0, 0, 0};
  }
}

// Picks a direction based on the BRDF
static vec3f sample_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
  // YOUR CODE GOES HERE
  if (material.roughness == 0) return {0, 0, 0};

  if (material.type == material_type::matte) {
    return sample_matte(material.color, normal, outgoing, rn);
  } else if (material.type == material_type::glossy) {
    return sample_glossy(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::reflective) {
    return sample_reflective(
        material.color, material.roughness, normal, outgoing, rn);
  } else if (material.type == material_type::transparent) {
    return sample_transparent(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::refractive) {
    return sample_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else {
    return {0, 0, 0};
  }
}

static vec3f sample_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl) {
  // YOUR CODE GOES HERE
  if (material.roughness != 0) return {0, 0, 0};

  if (material.type == material_type::reflective) {
    return sample_reflective(material.color, normal, outgoing);
  } else if (material.type == material_type::transparent) {
    return sample_transparent(
        material.color, material.ior, normal, outgoing, rnl);
  } else if (material.type == material_type::refractive) {
    return sample_refractive(
        material.color, material.ior, normal, outgoing, rnl);
  } else {
    return {0, 0, 0};
  }
}

// Compute the weight for sampling the BRDF
static float sample_bsdfcos_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  // YOUR CODE GOES HERE
  if (material.roughness == 0) return 0;

  if (material.type == material_type::matte) {
    return sample_matte_pdf(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::glossy) {
    return sample_glossy_pdf(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::reflective) {
    return sample_reflective_pdf(
        material.color, material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return sample_tranparent_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return sample_refractive_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else {
    return 0;
  }
}

static float sample_delta_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  // YOUR CODE GOES HERE
  if (material.roughness != 0) return 0;

  if (material.type == material_type::reflective) {
    return sample_reflective_pdf(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return sample_tranparent_pdf(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return sample_refractive_pdf(
        material.color, material.ior, normal, outgoing, incoming);
  } else {
    return 0;
  }
}

// Sample lights wrt solid angle
static vec3f sample_lights(const scene_data& scene,
    const pathtrace_lights& lights, const vec3f& position, float rl, float rel,
    const vec2f& ruv) {
  // YOUR CODE GOES HERE
  auto  light_id = sample_uniform((int)lights.lights.size(), rl);
  auto& light    = lights.lights[light_id];
  if (light.instance != invalidid) {
    auto& instance  = scene.instances[light.instance];
    auto& shape     = scene.shapes[instance.shape];
    auto  element   = sample_discrete(light.elements_cdf, rel);
    auto  uv        = (!shape.triangles.empty()) ? sample_triangle(ruv) : ruv;
    auto  lposition = eval_position(scene, instance, element, uv);
    return normalize(lposition - position);
  } else if (light.environment != invalidid) {
    auto& environment = scene.environments[light.environment];
    if (environment.emission_tex != invalidid) {
      auto& emission_tex = scene.textures[environment.emission_tex];
      auto  idx          = sample_discrete(light.elements_cdf, rel);
      auto  uv = vec2f{((idx % emission_tex.width) + 0.5f) / emission_tex.width,
          ((idx / emission_tex.width) + 0.5f) / emission_tex.height};
      return transform_direction(environment.frame,
          {cos(uv.x * 2 * pif) * sin(uv.y * pif), cos(uv.y * pif),
              sin(uv.x * 2 * pif) * sin(uv.y * pif)});
    } else {
      return sample_sphere(ruv);
    }
  } else {
    return {0, 0, 0};
  }
}

// Sample lights pdf
static float sample_lights_pdf(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const vec3f& position,
    const vec3f& direction) {
  // YOUR CODE GOES HERE
  auto pdf = 0.0f;
  for (auto& light : lights.lights) {
    if (light.instance != invalidid) {
      auto& instance      = scene.instances[light.instance];
      auto  lpdf          = 0.0f;
      auto  next_position = position;
      for (auto bounce = 0; bounce < 100; bounce++) {
        auto intersection = intersect_bvh(
            bvh, scene, light.instance, {next_position, direction});
        if (!intersection.hit) break;
        auto lposition = eval_position(
            scene, instance, intersection.element, intersection.uv);
        auto lnormal = eval_element_normal(
            scene, instance, intersection.element);
        auto area = light.elements_cdf.back();
        lpdf += distance_squared(lposition, position) /
                (abs(dot(lnormal, direction)) * area);
        next_position = lposition + direction * 1e-3f;
      }
      pdf += lpdf;
    } else if (light.environment != invalidid) {
      auto& environment = scene.environments[light.environment];
      if (environment.emission_tex != invalidid) {
        auto& emission_tex = scene.textures[environment.emission_tex];
        auto  wl = transform_direction(inverse(environment.frame), direction);
        auto  texcoord = vec2f{atan2(wl.z, wl.x) / (2 * pif),
            acos(clamp(wl.y, -1.0f, 1.0f)) / pif};
        if (texcoord.x < 0) texcoord.x += 1;
        auto i = clamp(
            (int)(texcoord.x * emission_tex.width), 0, emission_tex.width - 1);
        auto j    = clamp((int)(texcoord.y * emission_tex.height), 0,
            emission_tex.height - 1);
        auto prob = sample_discrete_pdf(
                        light.elements_cdf, j * emission_tex.width + i) /
                    light.elements_cdf.back();
        auto angle = (2 * pif / emission_tex.width) *
                     (pif / emission_tex.height) *
                     sin(pif * (j + 0.5f) / emission_tex.height);
        pdf += prob / angle;
      } else {
        pdf += 1 / (4 * pif);
      }
    }
  }
  pdf *= sample_uniform_pdf((int)lights.lights.size());
  return pdf;
}

struct pathtrace_result {
  vec3f radiance = {0, 0, 0};
  bool  hit      = false;
  vec3f albedo   = {0, 0, 0};
  vec3f normal   = {0, 0, 0};
};

static pathtrace_result denoise_shade_path(const scene_data& scene,
    const bvh_data& bvh, const pathtrace_lights& lights, const ray3f& ray_,
    rng_state& rng, const pathtrace_params& params) {
  auto radiance   = vec3f{0, 0, 0};
  auto weight     = vec3f{1, 1, 1};
  auto ray        = ray_;
  auto hit        = false;
  auto hit_albedo = vec3f{0, 0, 0};
  auto hit_normal = vec3f{0, 0, 0};
  auto opbounce   = 0;

  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    auto intersection = intersect_bvh(bvh, scene, ray);
    if (!intersection.hit) {
      radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce++ > 128) break;
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    radiance += weight * eval_emission(material, normal, outgoing);

    auto incoming = vec3f{0, 0, 0};
    if (!is_delta(material)) {
      if (rand1f(rng) < 0.5f) {
        incoming = sample_bsdfcos(
            material, normal, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
      }
      if (incoming == vec3f{0, 0, 0}) break;
      weight *=
          eval_bsdfcos(material, normal, outgoing, incoming) /
          (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));
    } else {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      weight *= eval_delta(material, normal, outgoing, incoming) /
                sample_delta_pdf(material, normal, outgoing, incoming);
    }

    ray = {position, incoming};

    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

static pathtrace_result denoise_shade_naive(const scene_data& scene,
    const bvh_data& bvh, const pathtrace_lights& lights, const ray3f& ray_,
    rng_state& rng, const pathtrace_params& params) {
  auto radiance   = vec3f{0, 0, 0};
  auto weight     = vec3f{1, 1, 1};
  auto ray        = ray_;
  auto hit        = false;
  auto hit_albedo = vec3f{0, 0, 0};
  auto hit_normal = vec3f{0, 0, 0};
  auto opbounce   = 0;

  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    auto intersection = intersect_bvh(bvh, scene, ray);
    if (!intersection.hit) {
      radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce++ > 128) break;
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    radiance += weight * eval_emission(material, normal, outgoing);

    auto incoming = vec3f{0, 0, 0};
    if (material.roughness != 0) {
      incoming = sample_bsdfcos(
          material, normal, outgoing, rand1f(rng), rand2f(rng));
      if (incoming == vec3f{0, 0, 0}) break;
      weight *= eval_bsdfcos(material, normal, outgoing, incoming) /
                sample_bsdfcos_pdf(material, normal, outgoing, incoming);
    } else {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      if (incoming == vec3f{0, 0, 0}) break;
      weight *= eval_delta(material, normal, outgoing, incoming) /
                sample_delta_pdf(material, normal, outgoing, incoming);
    }

    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }

    ray = {position, incoming};
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Recursive path tracing.
static vec4f shade_pathtrace(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray_, rng_state& rng,
    const pathtrace_params& params) {
  // YOUR CODE GOES HERE
  pathtrace_result result = denoise_shade_path(
      scene, bvh, lights, ray_, rng, params);

  vec3f radiance = result.radiance;

  return vec4f{radiance.x, radiance.y, radiance.z, result.hit ? 1.0f : 0.0f};
}

// Recursive path tracing.
static vec4f shade_naive(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray_, rng_state& rng,
    const pathtrace_params& params) {
  // YOUR CODE GOES HERE
  pathtrace_result result = denoise_shade_naive(
      scene, bvh, lights, ray_, rng, params);

  vec3f radiance = result.radiance;

  return vec4f{radiance.x, radiance.y, radiance.z, result.hit ? 1.0f : 0.0f};
}

// Eyelight for quick previewing.
static vec4f shade_eyelight(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray_, rng_state& rng,
    const pathtrace_params& params) {
  // initialize
  auto radiance = vec3f{0, 0, 0};
  auto weight   = vec3f{1, 1, 1};
  auto ray      = ray_;
  auto hit      = false;

  // trace  path
  for (auto bounce = 0; bounce < max(params.bounces, 4); bounce++) {
    // intersect next point
    auto intersection = intersect_bvh(bvh, scene, ray);
    if (!intersection.hit) {
      radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    // set hit variables
    if (bounce == 0) hit = true;

    // accumulate emission
    auto incoming = outgoing;
    radiance += weight * eval_emission(material, normal, outgoing);

    // brdf * light
    radiance += weight * pif *
                eval_bsdfcos(material, normal, outgoing, incoming);

    // continue path
    if (!is_delta(material)) break;
    incoming = sample_delta(material, normal, outgoing, rand1f(rng));
    if (incoming == vec3f{0, 0, 0}) break;
    weight *= eval_delta(material, normal, outgoing, incoming) /
              sample_delta_pdf(material, normal, outgoing, incoming);
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // setup next iteration
    ray = {position, incoming};
  }

  return {radiance.x, radiance.y, radiance.z, hit ? 1.0f : 0.0f};
}

// Normal for debugging.
static vec4f shade_normal(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray, rng_state& rng,
    const pathtrace_params& params) {
  // intersect next point
  auto intersection = intersect_bvh(bvh, scene, ray);
  if (!intersection.hit) return {0, 0, 0, 0};

  // prepare shading point
  auto outgoing = -ray.d;
  auto normal   = eval_shading_normal(scene, intersection, outgoing);
  return {normal.x, normal.y, normal.z, 1};
}

// Normal for debugging.
static vec4f shade_texcoord(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray, rng_state& rng,
    const pathtrace_params& params) {
  // intersect next point
  auto intersection = intersect_bvh(bvh, scene, ray);
  if (!intersection.hit) return {0, 0, 0, 0};

  // prepare shading point
  auto texcoord = eval_texcoord(scene, intersection);
  return {texcoord.x, texcoord.y, 0, 1};
}

// Color for debugging.
static vec4f shade_color(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray, rng_state& rng,
    const pathtrace_params& params) {
  // intersect next point
  auto intersection = intersect_bvh(bvh, scene, ray);
  if (!intersection.hit) return {0, 0, 0, 0};

  // prepare shading point
  auto color = eval_material(scene, intersection).color;
  return {color.x, color.y, color.z, 1};
}

#if YOCTO_DENOISE
// Trace a single ray from the camera using the given algorithm.
using pathtrace_shader_func = pathtrace_result (*)(const scene_data& scene,
    const bvh_scene& bvh, const pathtrace_lights& lights, const ray3f& ray,
    rng_state& rng, const pathtrace_params& params);
static pathtrace_shader_func get_shader(const pathtrace_params& params) {
  switch (params.shader) {
    case pathtrace_shader_type::pathtrace: return denoise_shade_path;
    case pathtrace_shader_type::naive: return denoise_shade_naive;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}
#else
// Trace a single ray from the camera using the given algorithm.
using pathtrace_shader_func = vec4f (*)(const scene_data& scene,
    const bvh_scene& bvh, const pathtrace_lights& lights, const ray3f& ray,
    rng_state& rng, const pathtrace_params& params);
static pathtrace_shader_func get_shader(const pathtrace_params& params) {
  switch (params.shader) {
    case pathtrace_shader_type::pathtrace: return shade_pathtrace;
    case pathtrace_shader_type::naive: return shade_naive;
    case pathtrace_shader_type::eyelight: return shade_eyelight;
    case pathtrace_shader_type::normal: return shade_normal;
    case pathtrace_shader_type::texcoord: return shade_texcoord;
    case pathtrace_shader_type::color: return shade_color;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}
#endif

// Build the bvh acceleration structure.
bvh_scene make_bvh(const scene_data& scene, const pathtrace_params& params) {
  return make_bvh(scene, false, false, params.noparallel);
}

// Init a sequence of random number generators.
pathtrace_state make_state(
    const scene_data& scene, const pathtrace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = pathtrace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image.assign(state.width * state.height, {0, 0, 0, 0});
  state.albedo.assign(state.width * state.height, {0, 0, 0});
  state.normal.assign(state.width * state.height, {0, 0, 0});
  state.hits.assign(state.width * state.height, 0);
  state.rngs.assign(state.width * state.height, {});
  auto rng_ = make_rng(1301081);
  for (auto& rng : state.rngs) {
    rng = make_rng(961748941ull, rand1i(rng_, 1 << 31) / 2 + 1);
  }
  return state;
}

// Init trace lights
pathtrace_lights make_lights(
    const scene_data& scene, const pathtrace_params& params) {
  auto lights = pathtrace_lights{};

  for (auto handle = 0; handle < scene.instances.size(); handle++) {
    auto& instance = scene.instances[handle];
    auto& material = scene.materials[instance.material];
    if (material.emission == vec3f{0, 0, 0}) continue;
    auto& shape = scene.shapes[instance.shape];
    if (shape.triangles.empty() && shape.quads.empty()) continue;
    auto& light       = lights.lights.emplace_back();
    light.instance    = handle;
    light.environment = invalidid;
    if (!shape.triangles.empty()) {
      light.elements_cdf = vector<float>(shape.triangles.size());
      for (auto idx = 0; idx < light.elements_cdf.size(); idx++) {
        auto& t                 = shape.triangles[idx];
        light.elements_cdf[idx] = triangle_area(
            shape.positions[t.x], shape.positions[t.y], shape.positions[t.z]);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
    if (!shape.quads.empty()) {
      light.elements_cdf = vector<float>(shape.quads.size());
      for (auto idx = 0; idx < light.elements_cdf.size(); idx++) {
        auto& t                 = shape.quads[idx];
        light.elements_cdf[idx] = quad_area(shape.positions[t.x],
            shape.positions[t.y], shape.positions[t.z], shape.positions[t.w]);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
  }
  for (auto handle = 0; handle < scene.environments.size(); handle++) {
    auto& environment = scene.environments[handle];
    if (environment.emission == vec3f{0, 0, 0}) continue;
    auto& light       = lights.lights.emplace_back();
    light.instance    = invalidid;
    light.environment = handle;
    if (environment.emission_tex != invalidid) {
      auto& texture      = scene.textures[environment.emission_tex];
      light.elements_cdf = vector<float>(texture.width * texture.height);
      for (auto idx = 0; idx < light.elements_cdf.size(); idx++) {
        auto ij    = vec2i{idx % texture.width, idx / texture.width};
        auto th    = (ij.y + 0.5f) * pif / texture.height;
        auto value = lookup_texture(texture, ij.x, ij.y);
        light.elements_cdf[idx] = max(value) * sin(th);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
  }

  // handle progress
  return lights;
}

// Progressively compute an image by calling trace_samples multiple times.
void pathtrace_sample(pathtrace_state& state, const scene_data& scene,
    const bvh_data& bvh, const pathtrace_lights& lights, int i, int j,
    const pathtrace_params& params) {
  auto& camera  = scene.cameras[params.camera];
  auto  sampler = get_shader(params);
  auto  idx     = state.width * j + i;
  auto  puv     = rand2f(state.rngs[idx]);
  auto  luv     = rand2f(state.rngs[idx]);
  auto  cuv     = vec2f{(i + puv.x) / state.width, (j + puv.y) / state.height};
  auto  ray     = eval_camera(camera, cuv, sample_disk(luv));
#if YOCTO_DENOISE
  auto [radiance, hit, albedo, normal] = sampler(
      scene, bvh, lights, ray, state.rngs[idx], params);
#else
  auto radiance = sampler(scene, bvh, lights, ray, state.rngs[idx], params);
  auto hit      = radiance.w;
#endif
  if (!isfinite(radiance)) radiance = {0, 0, 0};
  if (max(radiance) > params.clamp)
    radiance = radiance * (params.clamp / max(radiance));
  if (hit) {
    state.image[idx] += {radiance.x, radiance.y, radiance.z, 1};
#if YOCTO_DENOISE
    state.albedo[idx] += albedo;
    state.normal[idx] += normal;
#endif
    state.hits[idx] += 1;
  } else if (!scene.environments.empty()) {
    state.image[idx] += {radiance.x, radiance.y, radiance.z, 1};
#if YOCTO_DENOISE
    state.albedo[idx] += {1, 1, 1};
    state.normal[idx] += -ray.d;
#endif
    state.hits[idx] += 1;
  }
}

void pathtrace_samples(pathtrace_state& state, const scene_data& scene,
    const bvh_data& bvh, const pathtrace_lights& lights,
    const pathtrace_params& params) {
  if (state.samples >= params.samples) return;
  if (params.noparallel) {
    for (auto j = 0; j < state.height; j++) {
      for (auto i = 0; i < state.width; i++) {
        pathtrace_sample(state, scene, bvh, lights, i, j, params);
      }
    }
  } else {
    parallel_for(state.width, state.height, [&](int i, int j) {
      pathtrace_sample(state, scene, bvh, lights, i, j, params);
    });
  }
  state.samples += 1;
}

// Check image type
static void check_image(
    const color_image& image, int width, int height, bool linear) {
  if (image.width != width || image.height != height)
    throw std::invalid_argument{"image should have the same size"};
  if (image.linear != linear)
    throw std::invalid_argument{
        linear ? "expected linear image" : "expected srgb image"};
}

// Get resulting render
color_image get_render(const pathtrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_render(image, state);
  return image;
}
void get_render(color_image& image, const pathtrace_state& state) {
  check_image(image, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = state.image[idx] * scale;
  }
}

// Get denoised render
image_data get_denoised(const pathtrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_denoised(image, state);
  return image;
}
void get_denoised(image_data& image, const pathtrace_state& state) {
#if YOCTO_DENOISE
  // Create an Intel Open Image Denoise device
  oidn::DeviceRef device = oidn::newDevice();
  device.commit();

  // get image
  get_render(image, state);

  // get albedo and normal
  auto albedo = vector<vec3f>(image.pixels.size()),
       normal = vector<vec3f>(image.pixels.size());
  auto scale  = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    albedo[idx] = state.albedo[idx] * scale;
    normal[idx] = state.normal[idx] * scale;
  }

  // Create a denoising filter
  oidn::FilterRef filter = device.newFilter("RT");  // ray tracing filter
  filter.setImage("color", (void*)image.pixels.data(), oidn::Format::Float3,
      state.width, state.height, 0, sizeof(vec4f), sizeof(vec4f) * state.width);
  filter.setImage("albedo", (void*)albedo.data(), oidn::Format::Float3,
      state.width, state.height);
  filter.setImage("normal", (void*)normal.data(), oidn::Format::Float3,
      state.width, state.height);
  filter.setImage("output", image.pixels.data(), oidn::Format::Float3,
      state.width, state.height, 0, sizeof(vec4f), sizeof(vec4f) * state.width);
  filter.set("inputScale", 1.0f);  // set scale as fixed
  filter.set("hdr", true);         // image is HDR
  filter.commit();

  // Filter the image
  filter.execute();
#else
  get_render(image, state);
#endif
}

// Get denoising buffers
image_data get_albedo(const pathtrace_state& state) {
  auto albedo = make_image(state.width, state.height, true);
  get_albedo(albedo, state);
  return albedo;
}
void get_albedo(image_data& albedo, const pathtrace_state& state) {
  check_image(albedo, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    albedo.pixels[idx] = {state.albedo[idx].x * scale,
        state.albedo[idx].y * scale, state.albedo[idx].z * scale, 1.0f};
  }
}
image_data get_normal(const pathtrace_state& state) {
  auto normal = make_image(state.width, state.height, true);
  get_normal(normal, state);
  return normal;
}
void get_normal(image_data& normal, const pathtrace_state& state) {
  check_image(normal, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    normal.pixels[idx] = {state.normal[idx].x * scale,
        state.normal[idx].y * scale, state.normal[idx].z * scale, 1.0f};
  }
}

// Denoise image
image_data path_denoise_render(const image_data& render,
    const image_data& albedo, const image_data& normal) {
  auto denoised = make_image(render.width, render.height, render.linear);
  path_denoise_render(denoised, render, albedo, normal);
  return denoised;
}
void path_denoise_render(image_data& denoised, const image_data& render,
    const image_data& albedo, const image_data& normal) {
  check_image(denoised, render.width, render.height, render.linear);
  check_image(albedo, render.width, render.height, albedo.linear);
  check_image(normal, render.width, render.height, normal.linear);
#if YOCTO_DENOISE
  // Create an Intel Open Image Denoise device
  oidn::DeviceRef device = oidn::newDevice();
  device.commit();

  // set image
  denoised = render;

  // Create a denoising filter
  oidn::FilterRef filter = device.newFilter("RT");  // ray tracing filter
  filter.setImage("color", (void*)render.pixels.data(), oidn::Format::Float3,
      render.width, render.height, 0, sizeof(vec4f),
      sizeof(vec4f) * render.width);
  filter.setImage("albedo", (void*)albedo.pixels.data(), oidn::Format::Float3,
      albedo.width, albedo.height, 0, sizeof(vec4f),
      sizeof(vec4f) * albedo.width);
  filter.setImage("normal", (void*)normal.pixels.data(), oidn::Format::Float3,
      normal.width, normal.height, 0, sizeof(vec4f),
      sizeof(vec4f) * normal.width);
  filter.setImage("output", denoised.pixels.data(), oidn::Format::Float3,
      denoised.width, denoised.height, 0, sizeof(vec4f),
      sizeof(vec4f) * denoised.width);
  filter.set("inputScale", 1.0f);  // set scale as fixed
  filter.set("hdr", true);         // image is HDR
  filter.commit();

  // Filter the image
  filter.execute();
#else
  denoised = render;
#endif
}

}  // namespace yocto
