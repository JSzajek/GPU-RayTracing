#include "OpenCLUtils.h"
#include "OpenCVUtils.h"
#include "RandomUtils.h"
#include "Timer.h"

cl_device_id device = nullptr;
cl_context context = nullptr;
cl_program program = nullptr;
cl_kernel kernel = nullptr;
cl_command_queue queue = nullptr;
cl_int err = -1;

struct Vector4f
{
public:
	Vector4f() = default;

	Vector4f(float _x, float _y, float _z, float _w)
		: x(_x),
		y(_y),
		z(_z),
		w(_w)
	{
	}
public:
	float x = 0;
	float y = 0;
	float z = 0;
	float w = 0;
};

struct Material
{
	Vector4f diffuseColor;
	Vector4f specularColor;
	float shininess = 1;
	float reflectivity = 0;
	float pad = 0;
	float pad2 = 0;
};

struct Triangle
{
public:
	Vector4f vertex_0;
	Vector4f vertex_1;
	Vector4f vertex_2;
	int materialIndex = -1;
	int _padding[3];
};

int main()
{
	const int Width		= 1280;
	const int Height	= 720;

	// Camera setup
	Vector4f CameraPos(0.0f, 0.0f, 20.0f, 0.0f);
	Vector4f CameraDir(0.0f, 0.0f, -1.0f, 0.0f);
	float fov = 60.0f;

	std::vector<Material> Materials;
	Material mat1;
	mat1.diffuseColor = { 0.8f, 0.3f, 0.3f, 0 };
	mat1.specularColor = { 1.0f, 1.0f, 1.0f, 0 };
	mat1.reflectivity = 0.5f;
	mat1.shininess = 32.0f;
	Materials.emplace_back(mat1);

	Material mat2;
	mat2.diffuseColor = { 0.3f, 0.8f, 0.3f, 0 };
	mat2.specularColor = { 1.0f, 1.0f, 1.0f, 0 };
	mat2.reflectivity = 0.5f;
	mat2.shininess = 64.0f;
	Materials.emplace_back(mat2);

	Material mat3;
	mat3.diffuseColor = { 0.3f, 0.3f, 0.8f, 0 };
	mat3.specularColor = { 1.0f, 1.0f, 1.0f, 0 };
	mat3.reflectivity = 0.7f;
	mat3.shininess = 128.0f;
	Materials.emplace_back(mat3);


	std::vector<Triangle> Triangles;
	Triangle tri1;
	tri1.vertex_0 = {0, 0, 0, 0};
	tri1.vertex_1 = {1, 0, 0, 0};
	tri1.vertex_2 = {0, 1, 0, 0};
	tri1.materialIndex = 0;
	Triangles.emplace_back(tri1);

	Triangle tri2;
	tri2.vertex_0 = { 1, 0, 0, 0 };
	tri2.vertex_1 = { 1, 1, 0, 0 };
	tri2.vertex_2 = { 0, 1, 0, 0 };
	tri2.materialIndex = 1;
	Triangles.emplace_back(tri2);

	Triangle tri3;
	tri3.vertex_0 = { 0, 0, 0, 0 };
	tri3.vertex_1 = { -3, 0, 0, 0 };
	tri3.vertex_2 = { 0, -3, 0, 0 };
	tri3.materialIndex = 2;
	Triangles.emplace_back(tri3);

	std::vector<Vector4f> Lights;
	Lights.emplace_back(Vector4f(2.0f, 2.0f, -3.0f, 1.0f));

	if (!OpenCLUtils::initialize_device_and_context(device, context))
		return -1;

	if (!OpenCLUtils::initialize_program("shaders/tracing.cl", "trace", context, device, program, kernel, queue))
	{
		assert(false);
		return -1;
	}

	cv::Mat outputImg(Height, Width, CV_8UC4, cv::Scalar(0));

	const size_t imageBufferSize = Width * Height * sizeof(uint8_t) * 4;
	cl_mem imageBuffer = OpenCLUtils::create_inout_buffer(context, outputImg.data, imageBufferSize);

	int lightsCount = static_cast<int>(Lights.size());
	cl_mem lightsBuffer = OpenCLUtils::create_input_buffer(context, Lights.data(), lightsCount * sizeof(Vector4f));
	int materialsCount = static_cast<int>(Materials.size());
	cl_mem materialsBuffer = OpenCLUtils::create_input_buffer(context, Materials.data(), materialsCount * sizeof(Material));
	int trianglesCount = static_cast<int>(Triangles.size());
	cl_mem trianglesBuffer = OpenCLUtils::create_input_buffer(context, Triangles.data(), trianglesCount * sizeof(Triangle));

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
	err |= clSetKernelArg(kernel, 1, sizeof(int), &Width);
	err |= clSetKernelArg(kernel, 2, sizeof(int), &Height);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &lightsBuffer);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &lightsCount);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &trianglesBuffer);
	err |= clSetKernelArg(kernel, 6, sizeof(int), &trianglesCount);
	err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &materialsBuffer);
	err |= clSetKernelArg(kernel, 8, sizeof(Vector4f), &CameraPos);
	err |= clSetKernelArg(kernel, 9, sizeof(Vector4f), &CameraDir);
	err |= clSetKernelArg(kernel, 10, sizeof(float), &fov);
	if (err < 0)
	{
		perror("Couldn't create a kernel argument");
		return false;
	}

	size_t global[2] = { Width, Height };

	const std::string winName = "Triangle Tracing";
	cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
	cv::imshow(winName, outputImg);

	Timer gpuBufferReadTimer;
	Timer drawTimer;

	float deltaTime_s = 0.01f;
	while (true)
	{
		gpuBufferReadTimer.Start();

		err = clEnqueueNDRangeKernel(queue,
									 kernel,
									 2,
									 NULL,
									 (const size_t*)&global,
									 NULL,
									 0,
									 NULL,
									 NULL);

		if (err < 0)
		{
			perror("Couldn't enqueue the kernel");
			return false;
		}

		///* Read the kernel's output    */
		err = clEnqueueReadBuffer(queue,
								  imageBuffer,
								  CL_FALSE,
								  0,
								  imageBufferSize,
								  outputImg.data,
								  0,
								  NULL,
								  NULL);

		if (err < 0)
		{
			perror("Couldn't read the buffer");
			return false;
		}

		clFinish(queue);

		const double gpuBufferTime_ms = gpuBufferReadTimer.Elapsed_ms();

		// Visualization logic
		drawTimer.Start();

		cv::cvtColor(outputImg, outputImg, cv::COLOR_RGBA2BGRA);
		cv::imshow(winName, outputImg);

		// Press 'ESC' to exit
		if (cv::waitKey(1) == 27)
		{
			break;
		}

		const double drawTime_ms = drawTimer.Elapsed_ms();

		std::cout << "GPU Read Time: " << std::to_string(gpuBufferTime_ms) << "\tDraw Time: " << std::to_string(drawTime_ms) << std::endl;
		deltaTime_s = static_cast<float>(gpuBufferTime_ms + drawTime_ms) * 0.01f; // Convert back to seconds
	}
}