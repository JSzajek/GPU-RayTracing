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
	inline float operator[](int index)
	{
		if (index == 0)
			return x;
		if (index == 1)
			return y;
		if (index == 2)
			return z;
		if (index == 3)
			return w;
		
		throw std::exception("Out of Bounds Index!");
	}

	inline Vector4f operator+(const Vector4f& other) const
	{
		return { x + other.x,
				 y + other.y,
				 z + other.z,
				 w + other.w };
	}

	inline Vector4f operator-(const Vector4f& other) const
	{
		return { x - other.x,
				 y - other.y,
				 z - other.z,
				 w - other.w };
	}

	inline Vector4f operator/(float scalar) const
	{
		return { x / scalar,
				 y / scalar,
				 z / scalar,
				 w / scalar };
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
	Triangle() = default;

	Triangle(const Triangle&) = default;

	
public:
	inline Vector4f Centriod() const
	{
		return (vertex_0 + vertex_1 + vertex_2) / 3.0f;
	}
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