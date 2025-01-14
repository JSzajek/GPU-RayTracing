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

struct Vector2f
{
public:
	Vector2f() = default;

	Vector2f(float _x, float _y)
		: x(_x),
		y(_y)
	{
	}
public:
	inline float Length() const { return std::sqrt(x * x + y * y); }
public:
	float x = 0;
	float y = 0;
};

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

struct Sphere
{
public:
	Vector4f position;
	Vector4f color;
	Vector4f attribution;
	float radius = 1;
	float reflectivity = 0;
	float pad = 0;
	float pad2 = 0;
};

int main()
{
	const int Width		= 1280;
	const int Height	= 720;

	// Camera setup
	Vector4f CameraPos(0.0f, 0.0f, 10.0f, 0.0f);
	Vector4f CameraDir(0.0f, 0.0f, -1.0f, 0.0f);
	float fov = 60.0f;

	std::vector<Sphere> Spheres;

	Sphere sphere1;
	sphere1.position = { -0.5f, -1.0f, -5.0f, 0 };
	sphere1.color = { 1.0f, 0.0f, 0.0f, 0 };
	sphere1.attribution = { 1.0f, 1.0f, 1.0f, 0 };
	sphere1.radius = 1.7f;
	sphere1.reflectivity = 0.4f;
	Spheres.emplace_back(sphere1);

	Sphere sphere2;
	sphere2.position = { -4.0f, -0.5f, -5.0f, 0 };
	sphere2.color = { 0.0f, 1.0f, 0.0f, 0 };
	sphere2.attribution = { 1.0f, 1.0f, 1.0f, 0 };
	sphere2.radius = 0.8f;
	sphere2.reflectivity = 0.2f;
	Spheres.emplace_back(sphere2);

	Sphere sphere3;
	sphere3.position = { 3.5f, 1.5f, -5.0f, 0 };
	sphere3.color = { 1.0f, 0.0f, 1.0f, 0 };
	sphere3.attribution = { 1.0f, 1.0f, 1.0f, 0 };
	sphere3.radius = 2.25f;
	sphere3.reflectivity = 0.3f;
	Spheres.emplace_back(sphere3);


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
	int spheresCount = static_cast<int>(Spheres.size());
	cl_mem spheresBuffer = OpenCLUtils::create_input_buffer(context, Spheres.data(), spheresCount * sizeof(Sphere));

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
	err |= clSetKernelArg(kernel, 1, sizeof(int), &Width);
	err |= clSetKernelArg(kernel, 2, sizeof(int), &Height);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &lightsBuffer);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &lightsCount);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &spheresBuffer);
	err |= clSetKernelArg(kernel, 6, sizeof(int), &spheresCount);
	err |= clSetKernelArg(kernel, 7, sizeof(Vector4f), &CameraPos);
	err |= clSetKernelArg(kernel, 8, sizeof(Vector4f), &CameraDir);
	err |= clSetKernelArg(kernel, 9, sizeof(float), &fov);
	if (err < 0)
	{
		perror("Couldn't create a kernel argument");
		return false;
	}

	size_t global[2] = { Width, Height };

	const std::string winName = "Sphere Tracing";
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