#include "OpenCLUtils.h"
#include "OpenCVUtils.h"
#include "RandomUtils.h"
#include "Timer.h"

#include "MeshDefines.h"
#include "MeshImporter.h"

cl_device_id device = nullptr;
cl_context context = nullptr;
cl_program program = nullptr;
cl_kernel kernel = nullptr;
cl_command_queue queue = nullptr;
cl_int err = -1;

struct AABB
{
	Vector4f mMin;
	Vector4f mMax;
};

struct BVHNode
{
	AABB mBounds;
	int mLeft = -1;
	int mRight = -1;
	int mStart = 0;
	int mCount = 0;
};

void UploadMesh(const Mesh& mesh, std::vector<Triangle>& output)
{
	for (const Triangle& triangle : mesh.triangles)
	{
		Triangle tri = triangle;
		tri.materialIndex = mesh.materialIndex;
		output.emplace_back(tri);
	}
}

void InitializeScene(std::vector<Material>& materials,
					 std::vector<Triangle>& triangles)
{
	Material mat1;
	mat1.diffuseColor = { 0.0f, 0.8f, 0.8f, 0 };
	mat1.specularColor = { 0.5f, 0.5f, 0.5f, 0 };
	mat1.reflectivity = 0.1f;
	mat1.shininess = 0.0f;
	materials.emplace_back(mat1);

	Material mat2;
	mat2.diffuseColor = { 0.0f, 0.8f, 0.3f, 0 };
	mat2.specularColor = { 1.0f, 1.0f, 1.0f, 0 };
	mat2.reflectivity = 0.0f;
	mat2.shininess = 64.0f;
	materials.emplace_back(mat2);

	Material mat3;
	mat3.diffuseColor = { 0.0f, 0.3f, 0.8f, 0 };
	mat3.specularColor = { 1.0f, 1.0f, 1.0f, 0 };
	mat3.reflectivity = 0.3f;
	mat3.shininess = 128.0f;
	materials.emplace_back(mat3);

	Mesh suzanne;
	MeshImporter::Import("content/suzanne.obj", suzanne);
	suzanne.materialIndex = 0;

	Mesh sphere;
	MeshImporter::Import("content/sphere.obj", sphere);
	sphere.materialIndex = 2;

	Mesh plane;
	MeshImporter::Import("content/plane.obj", plane);
	plane.materialIndex = 1;

	UploadMesh(suzanne, triangles);
	UploadMesh(sphere, triangles);
	UploadMesh(plane, triangles);
}

Vector4f ComponentMinimum(const Vector4f& a, const Vector4f& b)
{
	return {std::min(a.x, b.x),
			std::min(a.y, b.y),
			std::min(a.z, b.z),
			std::min(a.w, b.w)};
}

Vector4f ComponentMaximum(const Vector4f& a, const Vector4f& b)
{
	return {std::max(a.x, b.x),
			std::max(a.y, b.y),
			std::max(a.z, b.z),
			std::max(a.w, b.w)};
}

AABB ComputeAABB(const std::vector<Triangle>& triangles, int start, int end) 
{
	AABB aabb;
	
	// Initialize AABB to very large and small values
	aabb.mMin = Vector4f(FLT_MAX, FLT_MAX, FLT_MAX, 0);
	aabb.mMax = Vector4f(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0);

	for (int i = start; i < end; ++i)
	{
		const Triangle& tri = triangles[i];

		aabb.mMin = ComponentMinimum(aabb.mMin, tri.vertex_0);
		aabb.mMin = ComponentMinimum(aabb.mMin, tri.vertex_1);
		aabb.mMin = ComponentMinimum(aabb.mMin, tri.vertex_2);

		aabb.mMax = ComponentMaximum(aabb.mMax, tri.vertex_0);
		aabb.mMax = ComponentMaximum(aabb.mMax, tri.vertex_1);
		aabb.mMax = ComponentMaximum(aabb.mMax, tri.vertex_2);
	}

	return aabb;
}

void ConstructBVH(std::vector<BVHNode>& nodes,
				  std::vector<Triangle>& triangles,
				  int maxTrianglesPerLeaf = 2)
{
	nodes.reserve(triangles.size() * 2);

	struct BuildTask
	{
		int mStart = 0;
		int mEnd = 0;
		int mNodeIndex = 0;
	};

	std::vector<BuildTask> stack;
	stack.push_back({0, static_cast<int>(triangles.size()), 0});

	nodes.push_back(BVHNode());

	while (!stack.empty())
	{
		BuildTask task = stack.back();
		stack.pop_back();

		const int start = task.mStart;
		const int end = task.mEnd;
		const int nodeIndex = task.mNodeIndex;
		const int count = end - start;

		BVHNode& node = nodes[nodeIndex];
		node.mBounds = ComputeAABB(triangles, start, end);
		if (count <= maxTrianglesPerLeaf)
		{
			node.mStart = start;
			node.mCount = count;
			node.mLeft = -1;
			node.mRight = -1;
			continue;
		}

		Vector4f size = node.mBounds.mMax - node.mBounds.mMin;
		int axis = (size.x > size.y && size.x > size.z) ? 0 : (size.y > size.z ? 1 : 2);

		// Partition
		int mid = (count > 1) ? static_cast<int>((start + end) / 2.0f) : 1;
		std::nth_element(triangles.begin() + start, triangles.begin() + mid, triangles.begin() + end,
						 [axis](const Triangle& a, const Triangle& b)
						 {
							return a.Centriod()[axis] < b.Centriod()[axis];
						 });

		// Create children w/ placeholders
		int leftChild = static_cast<int>(nodes.size());
		nodes.push_back(BVHNode());

		int rightChild = static_cast<int>(nodes.size());
		nodes.push_back(BVHNode());

		node = nodes[nodeIndex];
		node.mLeft = leftChild;
		node.mRight = rightChild;
		node.mStart = -1;
		node.mCount = -1;

		stack.push_back({mid, end, rightChild});
		stack.push_back({start, mid, leftChild});
	}
}

int main()
{
	const int Width		= 1280;
	const int Height	= 720;

	// Camera setup
	Vector4f CameraPos(0.0f, 2.0f, 8.0f, 0.0f);
	Vector4f CameraDir(0.0f, -0.3f, -1.0f, 0.0f);
	float fov = 60.0f;

	std::vector<Material> Materials;
	std::vector<Triangle> Triangles;
	std::vector<Vector4f> Lights;
	Lights.emplace_back(Vector4f(1.0f, 3.0f, 0.0f, 5.0f));

	InitializeScene(Materials, Triangles);

	std::vector<BVHNode> bvh;
	ConstructBVH(bvh, Triangles, 5);


	Vector4f ray_origin(1, -3, 2, 0);
	Vector4f ray_target(1.7f, -0.4f, 1, 0);



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

	const int lightsCount = static_cast<int>(Lights.size());
	cl_mem lightsBuffer = OpenCLUtils::create_input_buffer(context, Lights.data(), lightsCount * sizeof(Vector4f));
	const int materialsCount = static_cast<int>(Materials.size());
	cl_mem materialsBuffer = OpenCLUtils::create_input_buffer(context, Materials.data(), materialsCount * sizeof(Material));
	const int trianglesCount = static_cast<int>(Triangles.size());
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

	const std::string winName = "Mesh Tracing";
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