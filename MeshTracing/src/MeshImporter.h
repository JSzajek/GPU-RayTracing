#pragma once

#include "MeshDefines.h"

#include <filesystem>
#include <string>

class MeshImporter
{
public:
	static bool Import(const std::filesystem::path& path, 
					   Mesh& output);
};