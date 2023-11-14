// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>


#include "../include/opencv2/v4d/scene.hpp"
#include <iostream>
#include <assimp/postprocess.h>
#include <opencv2/calib3d.hpp>
#include <functional>

namespace cv {
namespace v4d {
namespace gl {

#include <opencv2/core.hpp>

cv::Vec3f cross(const cv::Vec3f& v1, const cv::Vec3f& v2) {
    return cv::Vec3f(v1[1] * v2[2] - v1[2] * v2[1],
                     v1[2] * v2[0] - v1[0] * v2[2],
                     v1[0] * v2[1] - v1[1] * v2[0]);
}

void releaseAssimpScene(const aiScene* scene) {
    if (scene) {
        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            delete[] scene->mMeshes[i]->mVertices;
            delete[] scene->mMeshes[i]->mNormals;
            for (unsigned int j = 0; j < scene->mMeshes[i]->mNumFaces; ++j) {
                delete[] scene->mMeshes[i]->mFaces[j].mIndices;
            }
            delete[] scene->mMeshes[i]->mFaces;
            delete scene->mMeshes[i];
        }

        delete[] scene->mMeshes;
        delete scene->mRootNode;
        delete scene;
    }
}

aiScene* createAssimpScene(std::vector<cv::Point3f>& vertices) {
    if (vertices.size() % 3 != 0) {
    	vertices.resize(vertices.size() / 3);
    }

    aiScene* scene = new aiScene();
    aiMesh* mesh = new aiMesh();

    // Set vertices
    mesh->mVertices = new aiVector3D[vertices.size()];
    for (size_t i = 0; i < vertices.size(); ++i) {
        mesh->mVertices[i] = aiVector3D(vertices[i].x, vertices[i].y, vertices[i].z);
    }
    mesh->mNumVertices = static_cast<unsigned int>(vertices.size());

    // Generate normals
    mesh->mNormals = new aiVector3D[mesh->mNumVertices];
    std::fill(mesh->mNormals, mesh->mNormals + mesh->mNumVertices, aiVector3D(0.0f, 0.0f, 0.0f));

    size_t numFaces = vertices.size() / 3;  // Assuming each face has 3 vertices
    mesh->mFaces = new aiFace[numFaces];
    mesh->mNumFaces = static_cast<unsigned int>(numFaces);

    for (size_t i = 0; i < numFaces; ++i) {
        aiFace& face = mesh->mFaces[i];
        face.mIndices = new unsigned int[3];  // Assuming each face has 3 vertices
        face.mIndices[0] = static_cast<unsigned int>(3 * i);
        face.mIndices[1] = static_cast<unsigned int>(3 * i + 1);
        face.mIndices[2] = static_cast<unsigned int>(3 * i + 2);
        face.mNumIndices = 3;

        // Calculate normal for this face
        aiVector3D edge1 = mesh->mVertices[face.mIndices[1]] - mesh->mVertices[face.mIndices[0]];
        aiVector3D edge2 = mesh->mVertices[face.mIndices[2]] - mesh->mVertices[face.mIndices[0]];
        aiVector3D normal = edge1 ^ edge2;  // Cross product
        normal.Normalize();

        // Assign the computed normal to all three vertices of the triangle
        mesh->mNormals[face.mIndices[0]] = normal;
        mesh->mNormals[face.mIndices[1]] = normal;
        mesh->mNormals[face.mIndices[2]] = normal;
    }

    // Attach the mesh to the scene
    scene->mMeshes = new aiMesh*[1];
    scene->mMeshes[0] = mesh;
    scene->mNumMeshes = 1;

    // Create a root node and attach the mesh
    scene->mRootNode = new aiNode();
    scene->mRootNode->mMeshes = new unsigned int[1]{0};
    scene->mRootNode->mNumMeshes = 1;

    return scene;
}

cv::Vec3f rotate3D(const cv::Vec3f& point, const cv::Vec3f& center, const cv::Vec3f& rotation)
{
    // Convert rotation vector to rotation matrix
    cv::Matx33f rotationMatrix;
    cv::Rodrigues(rotation, rotationMatrix);

    // Subtract center from point
    cv::Vec3f translatedPoint = point - center;

    // Rotate the point using the rotation matrix
    cv::Vec3f rotatedPoint = rotationMatrix * translatedPoint;

    // Translate the point back
    rotatedPoint += center;

    return rotatedPoint;
}

cv::Matx44f perspective(float fov, float aspect, float zNear, float zFar) {
    float tanHalfFovy = tan(fov / 2.0f);

    cv::Matx44f projection = cv::Matx44f::eye();
    projection(0, 0) = 1.0f / (aspect * tanHalfFovy);
    projection(1, 1) = 1.0f / (tanHalfFovy); // Invert the y-coordinate
    projection(2, 2) = -(zFar + zNear) / (zFar - zNear); // Invert the z-coordinate
    projection(2, 3) = -1.0f;
    projection(3, 2) = -(2.0f * zFar * zNear) / (zFar - zNear);
    projection(3, 3) = 0.0f;

    return projection;
}

cv::Matx44f lookAt(cv::Vec3f eye, cv::Vec3f center, cv::Vec3f up) {
    cv::Vec3f f = cv::normalize(center - eye);
    cv::Vec3f s = cv::normalize(f.cross(up));
    cv::Vec3f u = s.cross(f);

    cv::Matx44f view = cv::Matx44f::eye();
    view(0, 0) = s[0];
    view(0, 1) = u[0];
    view(0, 2) = -f[0];
    view(0, 3) = 0.0f;
    view(1, 0) = s[1];
    view(1, 1) = u[1];
    view(1, 2) = -f[1];
    view(1, 3) = 0.0f;
    view(2, 0) = s[2];
    view(2, 1) = u[2];
    view(2, 2) = -f[2];
    view(2, 3) = 0.0f;
    view(3, 0) = -s.dot(eye);
    view(3, 1) = -u.dot(eye);
    view(3, 2) = f.dot(eye);
    view(3, 3) = 1.0f;

    return view;
}

cv::Matx44f modelView(const cv::Vec3f& translation, const cv::Vec3f& rotationVec, const cv::Vec3f& scaleVec) {
    cv::Matx44f scaleMat(
    		scaleVec[0], 0.0, 0.0, 0.0,
            0.0, scaleVec[1], 0.0, 0.0,
            0.0, 0.0, scaleVec[2], 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotXMat(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos(rotationVec[0]), -sin(rotationVec[0]), 0.0,
            0.0, sin(rotationVec[0]), cos(rotationVec[0]), 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotYMat(
            cos(rotationVec[1]), 0.0, sin(rotationVec[1]), 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin(rotationVec[1]), 0.0,cos(rotationVec[1]), 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotZMat(
            cos(rotationVec[2]), -sin(rotationVec[2]), 0.0, 0.0,
            sin(rotationVec[2]), cos(rotationVec[2]), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f translateMat(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            translation[0], translation[1], translation[2], 1.0);

    return translateMat * rotXMat * rotYMat * rotZMat * scaleMat;
}


static void calculateBoundingBox(const aiMesh* mesh, cv::Vec3f& min, cv::Vec3f& max) {
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        aiVector3D vertex = mesh->mVertices[i];
        if (i == 0) {
            min = max = cv::Vec3f(vertex.x, vertex.y, vertex.z);
        } else {
            min[0] = std::min(min[0], vertex.x);
            min[1] = std::min(min[1], vertex.y);
            min[2] = std::min(min[2], vertex.z);

            max[0] = std::max(max[0], vertex.x);
            max[1] = std::max(max[1], vertex.y);
            max[2] = std::max(max[2], vertex.z);
        }
    }
}

static void calculateBoundingBoxInfo(const aiMesh* mesh, cv::Vec3f& center, cv::Vec3f& size) {
    cv::Vec3f min, max;
    calculateBoundingBox(mesh, min, max);
    center = (min + max) / 2.0f;
    size = max - min;
}

static float calculateAutoScale(const aiMesh* mesh) {
    cv::Vec3f center, size;
    calculateBoundingBoxInfo(mesh, center, size);

    float maxDimension = std::max(size[0], std::max(size[1], size[2]));
    return 1.0f / maxDimension;
}

static void drawMesh(aiMesh* mesh, Scene::RenderMode mode) {
    // Generate and bind VAO
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Load vertex data
    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(float), mesh->mVertices, GL_STATIC_DRAW);

    // Specify vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Load index data, if present
    if (mesh->HasFaces()) {
        std::vector<unsigned int> indices;
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }

        if (mode != Scene::RenderMode::DEFAULT) {
            // Duplicate vertices for wireframe rendering or point rendering
            std::vector<unsigned int> modifiedIndices;
            for (size_t i = 0; i < indices.size(); i += 3) {
                if (mode == Scene::RenderMode::WIREFRAME) {
                    // Duplicate vertices for wireframe rendering
                    modifiedIndices.push_back(indices[i]);
                    modifiedIndices.push_back(indices[i + 1]);

                    modifiedIndices.push_back(indices[i + 1]);
                    modifiedIndices.push_back(indices[i + 2]);

                    modifiedIndices.push_back(indices[i + 2]);
                    modifiedIndices.push_back(indices[i]);
                }

                if (mode == Scene::RenderMode::POINTCLOUD) {
                    // Duplicate vertices for point rendering
                    modifiedIndices.push_back(indices[i]);
                    modifiedIndices.push_back(indices[i + 1]);
                    modifiedIndices.push_back(indices[i + 2]);
                }
            }

            GLuint EBO;
            glGenBuffers(1, &EBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, modifiedIndices.size() * sizeof(unsigned int), &modifiedIndices[0], GL_STATIC_DRAW);

            // Draw as lines or points
            if (mode == Scene::RenderMode::WIREFRAME) {
                glDrawElements(GL_LINES, modifiedIndices.size(), GL_UNSIGNED_INT, 0);
            } else if (mode == Scene::RenderMode::POINTCLOUD) {
                glDrawElements(GL_POINTS, modifiedIndices.size(), GL_UNSIGNED_INT, 0);
            }

            // Cleanup
            glDeleteBuffers(1, &EBO);
        } else {
            GLuint EBO;
            glGenBuffers(1, &EBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

            // Draw as triangles
            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

            // Cleanup
            glDeleteBuffers(1, &EBO);
        }
    } else {
        glDrawArrays(GL_TRIANGLES, 0, mesh->mNumVertices);
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

// Function to recursively draw a node and its children
static void drawNode(aiNode* node, const aiScene* scene, Scene::RenderMode mode) {
    // Draw all meshes at this node
    for(unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        drawMesh(mesh, mode);
    }

    // Recurse for all children
    for(unsigned int i = 0; i < node->mNumChildren; i++) {
        drawNode(node->mChildren[i], scene, mode);
    }
}

// Function to draw a model
static void drawModel(const aiScene* scene, Scene::RenderMode mode) {
    // Draw the root node
    drawNode(scene->mRootNode, scene, mode);
}

static void applyModelView(cv::Mat_<float>& points, const cv::Matx44f& transformation) {
    // Ensure the input points matrix has the correct dimensions (3 columns for x, y, z)
    CV_Assert(points.cols == 3);

    // Construct the 4x4 transformation matrix with scaling


    // Convert points to homogeneous coordinates (add a column of ones)
    cv::hconcat(points, cv::Mat::ones(points.rows, 1, CV_32F), points);

    // Transpose the points matrix for multiplication
    cv::Mat pointsTransposed = points.t();

    // Apply the transformation
    cv::Mat transformedPoints = transformation * pointsTransposed;

    // Transpose back to the original orientation
    transformedPoints = transformedPoints.t();

    // Extract the transformed 3D points (excluding the fourth homogeneous coordinate)
    points = transformedPoints(cv::Rect(0, 0, 3, transformedPoints.rows)).clone();
}

static void applyModelView(std::vector<cv::Point3f>& points, const cv::Matx44f& transformation) {
    // Ensure the input points vector is not empty
    if (points.empty()) {
        std::cerr << "Error: Input points vector is empty.\n";
        return;
    }

    // Apply the model-view transformation to each point
    for (auto& point : points) {
        // Convert the point to a column vector
        cv::Mat pointMat = (cv::Mat_<float>(3, 1) << point.x, point.y, point.z);

        pointMat = transformation * pointMat;

        // Update the point with the transformed values
        point = cv::Point3f(pointMat.at<float>(0, 0), pointMat.at<float>(1, 0), pointMat.at<float>(2, 0));
    }
}

static void processNode(const aiNode* node, const aiScene* scene, cv::Mat_<float>& allVertices) {
    // Process all meshes in the current node
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // Process all vertices in the current mesh
        for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
            aiVector3D aiVertex = mesh->mVertices[j];
            cv::Mat_<float> vertex = (cv::Mat_<float>(1, 3) << aiVertex.x, aiVertex.y, aiVertex.z);
            allVertices.push_back(vertex);
        }
    }

    // Recursively process child nodes
    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        processNode(node->mChildren[i], scene, allVertices);
    }
}

static void processNode(const aiNode* node, const aiScene* scene, std::vector<cv::Point3f>& allVertices) {
    // Process all meshes in the current node
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // Process all vertices in the current mesh
        for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
            aiVector3D aiVertex = mesh->mVertices[j];
            cv::Point3f vertex(aiVertex.x, aiVertex.y, aiVertex.z);
            allVertices.push_back(vertex);
        }
    }

    // Recursively process child nodes
    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        processNode(node->mChildren[i], scene, allVertices);
    }
}

Scene::Scene() {
}

Scene::~Scene() {
}

void Scene::reset() {
	if(shaderHandles_[0] > 0)
		glDeleteProgram(shaderHandles_[0]);
	if(shaderHandles_[1] > 0)
		glDeleteShader(shaderHandles_[1]);
	if(shaderHandles_[2] > 0)
		glDeleteShader(shaderHandles_[2]);
	//FIXME how to cleanup a scene?
//	releaseAssimpScene(scene_);
}

bool Scene::load(const std::vector<Point3f>& points) {
	reset();
	std::vector<Point3f> copy = points;
    scene_ = createAssimpScene(copy);
    cv::v4d::initShader(shaderHandles_, vertexShaderSource_.c_str(), fragmentShaderSource_.c_str(), "fragColor");
    calculateBoundingBoxInfo(scene_->mMeshes[0], autoCenter_, size_);
    autoScale_ = calculateAutoScale(scene_->mMeshes[0]);
    return true;
}


bool Scene::load(const std::string& filename) {
	reset();
    scene_ = importer_.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenNormals);

    if (!scene_ || (scene_->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene_->mRootNode) {
        return false;
    }


    cv::v4d::initShader(shaderHandles_, vertexShaderSource_.c_str(), fragmentShaderSource_.c_str(), "fragColor");
    calculateBoundingBoxInfo(scene_->mMeshes[0], autoCenter_, size_);
    autoScale_ = calculateAutoScale(scene_->mMeshes[0]);
    return true;
}

cv::Mat_<float> Scene::pointCloudAsMat() {
	cv::Mat_<float> allVertices;
	processNode(scene_->mRootNode, scene_, allVertices);
	return allVertices;
}

std::vector<cv::Point3f> Scene::pointCloudAsVector() {
	std::vector<cv::Point3f> allVertices;
	processNode(scene_->mRootNode, scene_, allVertices);
	return allVertices;
}

void Scene::render(const cv::Rect& viewport, const cv::Matx44f& projection, const cv::Matx44f& view, const cv::Matx44f& modelView) {
	glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glUniformMatrix4fv(glGetUniformLocation(shaderHandles_[0], "projection"), 1, GL_FALSE, projection.val);
	glUniformMatrix4fv(glGetUniformLocation(shaderHandles_[0], "view"), 1, GL_FALSE, view.val);
	glUniform3fv(glGetUniformLocation(shaderHandles_[0], "lightPos"), 1, lightPos_.val);
	glUniform3fv(glGetUniformLocation(shaderHandles_[0], "viewPos"), 1, viewPos_.val);
	glUniform1i(glGetUniformLocation(shaderHandles_[0], "renderMode"), mode_);
	glUniformMatrix4fv(glGetUniformLocation(shaderHandles_[0], "model"), 1, GL_FALSE, modelView.val);
    glUseProgram(shaderHandles_[0]);

	drawModel(scene_, mode_);
}

} /* namespace gl */
} /* namespace v4d */
} /* namespace cv */
