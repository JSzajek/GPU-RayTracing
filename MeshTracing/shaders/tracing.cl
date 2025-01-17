#define EPSILON 0.001f

typedef struct
{
    float3 origin;
    float3 direction;
} Ray;

typedef struct
{
    float4 diffuse_color;
    float4 specular_color;
    float shininess;
    float reflectivity;
    float pad_0;
    float pad_1;
} Material;

typedef struct 
{
    float4 vertex_0;
    float4 vertex_1;
    float4 vertex_2;
    float4 normal_0;
    float4 normal_1;
    float4 normal_2;
    int materialIndex;
    int _padding[3];
} Triangle;

bool ray_triangle_intersect(Ray ray,
                            Triangle tri,
                            float* t,
                            float3* hit_point,
                            float3* hit_normal)
{
    // Compute edges of the tri and the determinant (using Möller–Trumbore algorithm)
    float3 e1 = (tri.vertex_1 - tri.vertex_0).xyz;
    float3 e2 = (tri.vertex_2 - tri.vertex_0).xyz;
    float3 h = cross(ray.direction, e2);
    float a = dot(e1, h);
    if (a > -1e-6 && a < 1e-6)
        return false;

    float f = 1.0f / a;
    float3 s = ray.origin - tri.vertex_0.xyz;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) 
        return false;

    float3 q = cross(s, e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) 
        return false;

    float det = f * dot(e2, q);
    if (det > 1e-6 && det < *t)
    {
        float w = 1 - u - v;

        *t = det;
        *hit_point = ray.origin + det * ray.direction;
        //*hit_normal = normalize(cross(e1, e2));
        *hit_normal = normalize(tri.normal_0.xyz * w + tri.normal_1.xyz * u + tri.normal_2.xyz * v);
        return true;
    }
    return false;
}

float3 reflect(float3 I, float3 N) 
{
    return I - 2.0f * dot(I, N) * N;
}

__kernel void trace(__global uchar4* image,
                    int width,
                    int height,
                    const __global float4* lights,
                    int num_lights,
                    const __global Triangle* triangles,
                    int num_triangles,
                    const __global Material* materials,
                    float4 camera_pos,
                    float4 camera_dir,
                    float fov) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) 
        return;

    // Compute normalized screen coordinates
    float aspect_ratio = (float)width / height;
    float px = (2.0f * ((x + 0.5f) / width) - 1.0f) * tan(radians(fov) / 2.0f) * aspect_ratio;
    float py = (1.0f - 2.0f * ((y + 0.5f) / height)) * tan(radians(fov) / 2.0f);

    Ray ray;

    // Ray direction
    ray.direction = normalize((float3)(px, py, -1.0f));

    // Initialize ray
    ray.origin = camera_pos.xyz;
    ray.direction = normalize(camera_dir.xyz + ray.direction);

    // Initialize color
    float3 color = (float3)(0.0f, 0.0f, 0.0f);
    float3 sky_color_top = (float3)(0.757f, 0.965f, 1.0f);
    float3 sky_color_bottom = (float3)(0.3f, 0.5f, 1.0f);

    // Energy carried by the ray
    float3 throughput = (float3)(1.0f, 1.0f, 1.0f);

    const int max_bounces = 3;
    for (int b = 0; b < max_bounces; ++b)
    {
        // Trace ray for sphere intersections
        float t_min = 1e20f;
        int hit_idx = -1;
        int material_idx = -1;

        float3 hit_normal;
        float3 hit_point;

        for (int i = 0; i < num_triangles; i++)
        {
            Triangle tri = triangles[i];
            if (ray_triangle_intersect(ray, tri, &t_min, &hit_point, &hit_normal))
            {
                hit_idx = i;
                material_idx = tri.materialIndex;
            }
        }


        // No intersection
        if (hit_idx == -1)
        {
            float a = 0.5f * (ray.direction.y + 1.0f);
            a = clamp(a, 0.0f, 1.0f);
            float3 sky_color = mix(sky_color_bottom, sky_color_top, a);
            color += throughput * sky_color;
            break;
        }
        
        // Material properties
        Material material = materials[material_idx];
        float3 diffuse_color = material.diffuse_color.rgb;
        float3 specular_color = material.specular_color.rgb;
        
        //color += diffuse_color * 0.1f;
        
        // Accumulate color from lights (basic Phong shading)
        float3 direct_light = (float3)(0.0f, 0.0f, 0.0f);
        for (int l = 0; l < num_lights; ++l) 
        {
            float4 light = lights[l];
            float3 light_pos = (float3)(light.x, light.y, light.z);
        
            float3 light_dir = normalize(light_pos - hit_point);
            float light_intensity = fmax(dot(hit_normal, light_dir), 0.0f);
        
            float3 view_dir = normalize(ray.origin - hit_point);
            float3 reflect_dir = normalize(reflect(-light_dir, hit_normal));
        
            // Diffuse shading (Lambertian)
            direct_light += diffuse_color * light_intensity * (float3)(1.0f, 1.0f, 1.0f); // White light
        
            // Specular shading (Phong reflection model)
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0f), material.shininess);
            direct_light += specular_color * (float3)(1.0f, 1.0f, 1.0f) /* White light */ * spec * 0.1f /*Specular intensity scaling*/;
        }

        float3 reflected_color = (float3)(0.0f, 0.0f, 0.0f);
        if (material.reflectivity > 0.0f)
        {
            ray.direction = reflect(ray.direction, hit_normal); // Reflect ray direction

            ray.origin = hit_point + EPSILON * hit_normal; // Move slightly off the surface
        
            reflected_color = throughput * material.reflectivity;
        }
        
        // Blend colors based on reflectivity
        color += throughput * (((1.0f - material.reflectivity) * direct_light) + (material.reflectivity * reflected_color));

        // Scale throughput by remaining reflectivity
        throughput *= material.reflectivity;
        
        // If throughput becomes negligible, terminate early
        if (length(throughput) < EPSILON)
        {
            break;
        }
    }

    // Write to image
    image[y * width + x] = (uchar4)((uchar)(clamp(color.x, 0.0f, 1.0f) * 255),
                                    (uchar)(clamp(color.y, 0.0f, 1.0f) * 255),
                                    (uchar)(clamp(color.z, 0.0f, 1.0f) * 255), 255);
}
