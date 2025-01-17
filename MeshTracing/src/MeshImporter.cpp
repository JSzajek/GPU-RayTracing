#include "MeshImporter.h"

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "assimp/material.h"

#include <iostream>

bool MeshImporter::Import(const std::filesystem::path& path, 
						  Mesh& output)
{
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path.generic_string(),
											 aiProcessPreset_TargetRealtime_Quality |
											 aiProcess_FlipWindingOrder);

	const std::string extension = path.extension().generic_string();
	const std::filesystem::path currentDir = path.parent_path();

	if (!scene)
	{
		std::cerr << "Failed Read File: " << path << std::endl;
		return false;
	}

	if (!scene->HasMeshes())
	{
		std::cerr << "No Meshes In: " << path << std::endl;
		return false;
	}

	for (uint32_t meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
	{
		const aiMesh* mesh = scene->mMeshes[meshIndex];
		if (!mesh)
		{
			std::cerr << "Skipping Invalid Mesh At Index " << meshIndex << " in " << path << std::endl;
			continue;
		}

		/// Find the associated node when possible --------------
		aiNode* node = nullptr;
		for (uint32_t nodeIndex = 0; scene->mRootNode->mNumChildren; ++nodeIndex)
		{
			aiNode* childNode = scene->mRootNode->mChildren[nodeIndex];
			for (uint32_t i = 0; i < childNode->mNumMeshes; ++i)
			{
				if (childNode->mMeshes[i] == meshIndex)
				{
					node = childNode;
					break;
				}
			}

			if (node)
				break;
		}
		/// -----------------------------------------------------

		// Extract vertices
		std::vector<Vector4f> vertices;
		std::vector<Vector4f> normals;
		vertices.reserve(mesh->mNumVertices);
		normals.reserve(mesh->mNumVertices);
		for (uint32_t i = 0; i < mesh->mNumVertices; ++i)
		{
			aiVector3D& ai_vertex = mesh->mVertices[i];

			Vector4f vertex = Vector4f(ai_vertex.x,
									   ai_vertex.y,
									   ai_vertex.z, 0);
			
			vertices.emplace_back(vertex);

			if (mesh->HasNormals())
			{
				aiVector3D& ai_normal = mesh->mNormals[i];

				Vector4f normal = Vector4f(ai_normal.x,
										   ai_normal.y,
										   ai_normal.z, 0);

				normals.emplace_back(normal);
			}
		}
		

		// Extract faces
		for (uint32_t i = 0; i < mesh->mNumFaces; ++i)
		{
			const aiFace& face = mesh->mFaces[i];

			Triangle triangle;

			int32_t index_0 = face.mIndices[0];
			int32_t index_1 = face.mIndices[1];
			int32_t index_2 = face.mIndices[2];

			triangle.vertex_0 = vertices[index_0];
			triangle.vertex_1 = vertices[index_1];
			triangle.vertex_2 = vertices[index_2];

			triangle.normal_0 = normals[index_0];
			triangle.normal_1 = normals[index_1];
			triangle.normal_2 = normals[index_2];

			output.triangles.emplace_back(triangle);
		}
	}

	return false;
}
