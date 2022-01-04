//
// Implementation for Yocto/RayTrace.
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

#include "yocto_raytrace.h"

#include <yocto/yocto_cli.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_shading.h>
#include <yocto/yocto_shape.h>

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR SCENE EVALUATION
// -----------------------------------------------------------------------------
namespace yocto {

// Generates a ray from a camera for yimg::image plane coordinate uv and
// the lens coordinates luv.
static ray3f eval_camera(const camera_data& camera, const vec2f& uv) {
  // YOUR CODE GOES HERE
  return camera_ray(camera.frame, camera.lens, camera.aspect, camera.film, uv);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace yocto {

// EXTRA CREDIT SWITCHES - START
auto MATCAP = false;  // Enables matcap shading for plastic bunnies
// EXTRA CREDIT SWITCHES - END

// Raytrace toon renderer.
static vec4f shade_toon(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE

  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return rgb_to_rgba(eval_environment(scene, ray.d));

  auto& instance = scene.instances[isec.instance];
  auto& material = scene.materials[instance.material];
  auto& shape    = scene.shapes[instance.shape];
  auto  normal   = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));

  auto position = transform_point(
      instance.frame, eval_position(shape, isec.element, isec.uv));

  if (dot(normal, -ray.d) < 0) normal = -normal;

  auto textcoord = eval_texcoord(shape, isec.element, isec.uv);

  auto radiance = vec4f{
      material.emission.x, material.emission.y, material.emission.z, 1};

  auto color = rgb_to_rgba(material.color) *
               eval_texture(scene, material.color_tex, textcoord, true);

  if (eval_material(scene, instance, isec.element, isec.uv).opacity < 1) {
    color = rgb_to_rgba(
        material.color *
        xyz(eval_texture(scene, material.color_tex, textcoord, true)));

    auto opacity = material.opacity *
                   eval_texture(scene, material.color_tex, textcoord, true).w;

    if (rand1f(rng) < 1 - opacity)
      return (shade_toon(
          scene, bvh, ray3f{position, ray.d}, bounce + 1, rng, params));
  }
  if (bounce >= params.bounces) return radiance;

  auto source = vec3f{5, 120, 50};

  float NdotL = dot(source, normal);

  auto lightIntensity        = NdotL > 0 ? 1 : 0.f;
  lightIntensity             = smoothstep(0.0f, 0.2f, NdotL);
  vec3f lightintensityVector = vec3f{
      lightIntensity, lightIntensity, lightIntensity};

  lightintensityVector *= vec3f{0.7, 0.6, 0.9};

  lightintensityVector += vec3f{0.7, 0.6, 0.9};

  auto glossy  = 32;
  auto viewdir = normalize(sample_hemisphere(normal, rand2f(rng)));

  auto  halfway = normalize(source + viewdir);
  float NdotH   = dot(normal, halfway);

  auto specularIntensity = pow(lightIntensity * NdotH, float(glossy * glossy));

  auto  ss     = smoothstep(0.005f, 0.01f, specularIntensity) * 0.09f;
  float rimDot = 0.9f - dot(-ray.d, normal) * pow(NdotL, 0.05f);

  vec4f rimColor  = {1, 1, 1, 1};
  float rimAmount = 0.716;

  float rimIntensity = smoothstep(
      float(rimAmount - 0.01f), float(rimAmount + 0.01f), rimDot);

  auto rim     = rimIntensity * rimColor;
  auto new_ray = ray;
  new_ray.d    = -new_ray.d;

  auto isec2 = intersect_bvh(bvh, scene, ray3f{position, source - position});

  if (dot(normal, normalize(source - position)) > 0 && isec2.hit) {
    return color *
           (rgb_to_rgba(
                vec3f{lightintensityVector.x + ss, lightintensityVector.y + ss,
                    lightintensityVector.z + ss} +
                specularIntensity) +
               rim) *
           0.8;
  }

  return color * (rgb_to_rgba(vec3f{lightintensityVector.x + ss,
                                  lightintensityVector.y + ss,
                                  lightintensityVector.z + ss} +
                              specularIntensity) +
                     rim);
}

// Raytrace renderer.
static vec4f shade_raytrace(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE

  auto isec = intersect_bvh(bvh, scene, ray);

  if (!isec.hit) return rgb_to_rgba(eval_environment(scene, ray.d));

  auto& instance = scene.instances[isec.instance];

  auto& shape    = scene.shapes[instance.shape];
  auto& material = scene.materials[instance.material];

  vec3f outgoing = -ray.d;

  auto normal = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));

  if (!shape.lines.empty())
    normal = orthonormalize(outgoing, normal);
  else if (!shape.triangles.empty()) {
    if (dot(outgoing, normal) < 0) normal = -normal;
  }

  auto position = transform_point(
      instance.frame, eval_position(shape, isec.element, isec.uv));

  auto texcoords = eval_texcoord(shape, isec.element, isec.uv);

  auto radiance =
      material.emission *
      xyz(eval_texture(scene, material.emission_tex, texcoords, true));

  auto color_tex = eval_texture(scene, material.color_tex, texcoords, true);
  auto color_shp = eval_color(scene, instance, isec.element, isec.uv);

  auto color = material.color * xyz(color_tex) * xyz(color_shp);

  auto opacity = material.opacity * color_tex.w;

  if (rand1f(rng) < 1 - opacity)
    return shade_raytrace(
        scene, bvh, ray3f{position, -outgoing}, bounce + 1, rng, params);

  if (bounce >= params.bounces)
    return vec4f{radiance.x, radiance.y, radiance.z, 1};

  vec3f incoming, halfway;
  vec4f calc;

  switch (material.type) {
    case material_type::matte:
      incoming = sample_hemisphere_cos(normal, rand2f(rng));
      radiance += color *
                  rgba_to_rgb(shade_raytrace(scene, bvh,
                      ray3f{position, incoming}, bounce + 1, rng, params));
      break;
    case material_type::transparent:
      if (rand1f(rng) <
          fresnel_schlick({0.04, 0.04, 0.04}, normal, outgoing).x) {
        incoming = reflect(outgoing, normal);
        calc     = shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
        radiance += {calc.x, calc.y, calc.z};
      } else {
        incoming = -outgoing;
        calc     = shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
        radiance += color * vec3f{calc.x, calc.y, calc.z};
      }
      break;
    case material_type::reflective:
      if (material.roughness > 0) {
        halfway = sample_hemisphere_cospower(
            2.f / (pow(material.roughness, 4.0f)), normal, rand2f(rng));
        incoming = reflect(outgoing, halfway);
        calc     = shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
        radiance += fresnel_schlick(color, halfway, outgoing) *
                    vec3f{calc.x, calc.y, calc.z};
      } else {
        incoming = reflect(outgoing, normal);
        calc     = shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
        radiance += fresnel_schlick(color, normal, outgoing) *
                    vec3f{calc.x, calc.y, calc.z};
      }
      break;
    case material_type::glossy:
      if (material.roughness > 0) {
        normal = sample_hemisphere_cospower(
            2.f / (pow(material.roughness, 4.0f)), normal, rand2f(rng));
      }
      if (rand1f(rng) >=
          fresnel_schlick({0.04, 0.04, 0.04}, normal, outgoing).x) {
        if (MATCAP) {
          incoming = reflect(outgoing, normal);
        } else {
          incoming = sample_hemisphere_cos(normal, rand2f(rng));
        }
        radiance += color *
                    rgba_to_rgb(shade_raytrace(scene, bvh,
                        ray3f{position, incoming}, bounce + 1, rng, params));
      } else {
        incoming = reflect(outgoing, normal);
        calc     = shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
        radiance += vec3f{calc.x, calc.y, calc.z};
      }
      break;
    default:
      incoming = sample_hemisphere(normal, rand2f(rng));
      calc     = shade_raytrace(
          scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
      radiance += (2 * pi) * color / pi * vec3f{calc.x, calc.y, calc.z} *
                  dot(normal, incoming);
      break;
  }

  return vec4f{radiance.x, radiance.y, radiance.z, material.opacity};
}

// Matte renderer.
static vec4f shade_matte(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE
  return {0, 0, 0, 0};
}

// Eyelight for quick previewing.
static vec4f shade_eyelight(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE
  auto isec = intersect_bvh(bvh, scene, ray);

  if (!isec.hit) return {0, 0, 0, 1};

  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];
  auto  normal   = eval_normal(shape, isec.element, isec.uv);
  auto& material = scene.materials[instance.material];
  auto  result   = material.color * dot(normal, -ray.d);

  return vec4f{result.x, result.y, result.z, 1};
}

static vec4f shade_normal(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return {0, 0, 0, 1};

  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];

  auto normal = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv) * 0.5 + 0.5);

  return rgb_to_rgba(normal);
}

static vec4f shade_texcoord(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return {0, 0, 0, 1};

  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];

  auto texcoord = eval_texcoord(shape, isec.element, isec.uv);
  return vec4f{fmod(texcoord.x, 1), fmod(texcoord.y, 1), 0, 0};
}

static vec4f shade_color(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) {
    return vec4f{0, 0, 0, 1};
  }
  auto  instance = scene.instances[isec.instance];
  auto& material = scene.materials[instance.material];

  return vec4f{
      material.color.x, material.color.y, material.color.z, material.opacity};
}

// Trace a single ray from the camera using the given algorithm.
using raytrace_shader_func = vec4f (*)(const scene_data& scene,
    const bvh_scene& bvh, const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params);
static raytrace_shader_func get_shader(const raytrace_params& params) {
  switch (params.shader) {
    case raytrace_shader_type::raytrace: return shade_raytrace;
    case raytrace_shader_type::matte: return shade_matte;
    case raytrace_shader_type::eyelight: return shade_eyelight;
    case raytrace_shader_type::normal: return shade_normal;
    case raytrace_shader_type::texcoord: return shade_texcoord;
    case raytrace_shader_type::color: return shade_color;
    case raytrace_shader_type::toon: return shade_toon;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}

// Build the bvh acceleration structure.
bvh_scene make_bvh(const scene_data& scene, const raytrace_params& params) {
  return make_bvh(scene, false, false, params.noparallel);
}

// Init a sequence of random number generators.
raytrace_state make_state(
    const scene_data& scene, const raytrace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = raytrace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image.assign(state.width * state.height, {0, 0, 0, 0});
  state.hits.assign(state.width * state.height, 0);
  state.rngs.assign(state.width * state.height, {});
  auto rng_ = make_rng(1301081);
  for (auto& rng : state.rngs) {
    rng = make_rng(961748941ull, rand1i(rng_, 1 << 31) / 2 + 1);
  }
  return state;
}

void raytrace_sample(raytrace_state& state, const scene_data& scene,
    const int i, const int j, const bvh_scene& bvh,
    const raytrace_params& params) {
  auto  idx = (state.width * j) + i;
  auto  puv = rand2f(state.rngs[idx]);
  vec2f uv  = vec2f{
      ((float)i + puv.x) / state.width, ((float)j + puv.y) / state.height};
  auto& camera = scene.cameras[params.camera];
  auto  ray    = eval_camera(camera, uv);
  auto  shader = get_shader(params);
  state.image[idx] += shader(scene, bvh, ray, 0, state.rngs[idx], params);
}

// Progressively compute an image by calling raytrace_samples multiple times.
void raytrace_samples(raytrace_state& state, const scene_data& scene,
    const bvh_scene& bvh, const raytrace_params& params) {
  if (state.samples >= params.samples) return;
  // YOUR CODE GOES HERE
  if (params.noparallel) {
    for (auto j : range(state.height)) {
      for (auto i : range(state.width)) {
        raytrace_sample(state, scene, i, j, bvh, params);
      }
    }
  } else {
    parallel_for(state.width, state.height, [&](int i, int j) {
      raytrace_sample(state, scene, i, j, bvh, params);
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
color_image get_render(const raytrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_render(image, state);
  return image;
}
void get_render(color_image& image, const raytrace_state& state) {
  check_image(image, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = state.image[idx] * scale;
  }
}

}  // namespace yocto
