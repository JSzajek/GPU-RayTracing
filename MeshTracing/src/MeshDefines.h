#pragma once

#include <vector>

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
	Vector4f normal_0;
	Vector4f normal_1;
	Vector4f normal_2;
	int materialIndex = -1;
	int _padding[3];
};

struct Mesh
{
	std::vector<Triangle> triangles;
	int materialIndex = -1;
};