#define EPSILON 0.001f

typedef struct
{
    float4 center;
    float4 color;
    float4 attribution;
    float radius;
    float reflectivity;
    float pad1;
    float pad2;
} Sphere;

bool ray_sphere_intersect(float3 ray_origin, 
                          float3 ray_dir, 
                          float3 sphere_center, 
                          float sphere_radius, 
                          float* t)
{
    float3 oc = ray_origin - sphere_center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
        float sqrt_d = sqrt(discriminant);
        float t1 = (-b - sqrt_d) / (2.0f * a);
        float t2 = (-b + sqrt_d) / (2.0f * a);
        *t = t1 > 0 ? t1 : t2; // Use the closest valid intersection
        return *t > 0;
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
                    const __global Sphere* spheres,
                    int num_spheres,
                    float4 camera_pos,
                    float4 camera_dir,
                    float fov) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) 
        return;
    
    if (get_global_id(0) == 0)
    {
        // Example debug output to confirm num_lights value
        //printf("Num Lights: %d\n", num_lights);
        //printf("Num Spheres: %d\n", num_spheres);

        //printf("Camera: (%f, %f, %f)   %f\n", camera_pos.x, camera_pos.y, camera_pos.z, fov);
    }

    // Compute normalized screen coordinates
    float aspect_ratio = (float)width / height;
    float px = (2.0f * ((x + 0.5f) / width) - 1.0f) * tan(radians(fov) / 2.0f) * aspect_ratio;
    float py = (1.0f - 2.0f * ((y + 0.5f) / height)) * tan(radians(fov) / 2.0f);

    // Ray direction
    float3 ray_direction = normalize((float3)(px, py, -1.0f));

    // Initialize ray
    float3 ray_origin = camera_pos.xyz;
    ray_direction = normalize(camera_dir.xyz + ray_direction);

    // Initialize color
    float3 color = (float3)(0.0f, 0.0f, 0.0f);
    float3 sky_color_top = (float3)(0.757f, 0.965f, 1.0f);
    float3 sky_color_bottom = (float3)(0.0f, 0.0f, 0.0f);

    // Energy carried by the ray
    float3 throughput = (float3)(1.0f, 1.0f, 1.0f);

    const int max_bounces = 2;
    for (int b = 0; b < max_bounces; ++b)
    {
        // Trace ray for sphere intersections
        float t_min = 1e20f;
        int hit_sphere_idx = -1;

        float3 hit_normal, hit_point;

        for (int i = 0; i < num_spheres; i++)
        {
            Sphere sphere = spheres[i];
            float3 sphereCenter = sphere.center.xyz;
            float sphereRadius = sphere.radius;

            //if (get_global_id(0) == 0)
            //{
            //    // Example debug output to confirm num_lights value
            //    printf("Sphere: %d rad: %f  (%f, %f, %f)\n", i, sphereRadius, sphereCenter.x, sphereCenter.y, sphereCenter.z);
            //}

            float t_intersect = 0;
            if (ray_sphere_intersect(ray_origin, ray_direction, sphereCenter, sphereRadius, &t_intersect))
            {
                if (t_intersect > 0.0f && t_intersect < t_min)
                {
                    t_min = t_intersect;
                    hit_sphere_idx = i;

                    hit_point = ray_origin + t_min * ray_direction;
                    hit_normal = normalize(hit_point - sphereCenter);
                }
            }
        }

        // No intersection
        if (hit_sphere_idx == -1)
        {
            float a = 0.5f * (ray_direction.y + 1.0f);
            a = clamp(a, 0.0f, 1.0f);
            float3 sky_color = mix(sky_color_bottom, sky_color_top, a);
            color += throughput * sky_color;
            break;
        }

        // Material properties
        Sphere sphere = spheres[hit_sphere_idx];
        float4 sphere_color = sphere.color;
        float sphere_reflectivity = sphere.reflectivity;

        // Accumulate color from lights (basic Phong shading)
        float3 direct_light = (float3)(0.0f, 0.0f, 0.0f);
        for (int l = 0; l < num_lights; ++l) 
        {
            float4 light = lights[l];
            float3 light_pos = (float3)(light.x, light.y, light.z);

            float3 light_dir = normalize(light_pos - hit_point);
            float light_intensity = fmax(dot(hit_normal, light_dir), 0.0f);

            //color += sphere_color.rgb * light_intensity * throughput;
            direct_light += light_intensity * (float3)(1.0f, 1.0f, 1.0f); // White light
        }

        float3 reflected_color = (float3)(0.0f, 0.0f, 0.0f);
        if (sphere_reflectivity > 0.0f)
        {
            ray_direction = reflect(ray_direction, hit_normal); // Reflect ray direction
            ray_origin = hit_point + EPSILON * hit_normal; // Move slightly off the surface

            reflected_color = throughput * sphere_color.rgb;
        }

        // Blend colors based on reflectivity
        float3 diffuse_color = sphere_color.rgb * direct_light;
        color += throughput * mix(diffuse_color, reflected_color, sphere_reflectivity);

        // Scale throughput by remaining reflectivity
        throughput *= sphere_reflectivity;

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

    // Write to image
    //image[y * width + x] = (uchar4)((uchar)(255),
    //                                (uchar)(0),
    //                                (uchar)(0), 
    //                                255);
}
