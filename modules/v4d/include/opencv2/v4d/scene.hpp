// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef MODULES_V4D_SRC_SCENE_HPP_
#define MODULES_V4D_SRC_SCENE_HPP_

#include "v4d.hpp"
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <string>

namespace cv {
namespace v4d {
namespace gl {

cv::Vec3f rotate3D(const cv::Vec3f& point, const cv::Vec3f& center, const cv::Vec3f& rotation);
cv::Matx44f perspective(float fov, float aspect, float zNear, float zFar);
cv::Matx44f lookAt(cv::Vec3f eye, cv::Vec3f center, cv::Vec3f up);
cv::Matx44f modelView(const cv::Vec3f& translation, const cv::Vec3f& rotationVec, const cv::Vec3f& scaleVec);

class Scene {
public:
	enum RenderMode {
		DEFAULT = 0,
		WIREFRAME = 1,
		POINTCLOUD = 2,
	};
private:
    Assimp::Importer importer_;
	const aiScene* scene_ = nullptr;
	RenderMode mode_ = DEFAULT;
	GLuint shaderHandles_[3] = {0, 0, 0};
	cv::Vec3f lightPos_ = {1.2f, 1.0f, 2.0f};
	cv::Vec3f viewPos_ = {0.0, 0.0, 0.0};

    cv::Vec3f autoCenter_, size_;
    float autoScale_ = 1;

    const string vertexShaderSource_ = R"(
 	    #version 300 es
 	    layout(location = 0) in vec3 aPos;
 	    out vec3 fragPos;
 	    uniform mat4 model;
 	    uniform mat4 view;
 	    uniform mat4 projection;
 	    void main() {
 	        gl_Position = projection * view * model * vec4(aPos, 1.0);
 	        fragPos = vec3(model * vec4(aPos, 1.0));
 	        gl_PointSize = 3.0;  // Set the size_ of the points
 	    }
 	)";


 	const string fragmentShaderSource_ = R"(
#version 300 es

#define RENDER_MODE_WIREFRAME 1
#define RENDER_MODE_POINTCLOUD 2

#define AMBIENT_COLOR vec3(0.95, 0.95, 0.95)
#define DIFFUSE_COLOR vec3(0.8, 0.8, 0.8)
#define SPECULAR_COLOR vec3(0.7, 0.7, 0.7)

// Control defines for effects
#define ENABLE_HDR true
#define HDR_EXPOSURE 1.0

#define ENABLE_BLOOM true
#define BLOOM_INTENSITY 1.0

#define ENABLE_SHADOWS true

precision highp float;

in vec3 fragPos;
out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform int renderMode;

// Function to check ray-sphere intersection
bool intersectSphere(vec3 rayOrigin, vec3 rayDir, vec3 sphereCenter, float sphereRadius) {
    vec3 oc = rayOrigin - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - 4.0 * a * c;
    return (discriminant > 0.0);
}

// Function to check if a point is in shadow
bool isInShadow(vec3 fragPos, vec3 lightDir) {
    // Use ray tracing to check for shadows (sphere example)
    vec3 rayOrigin = fragPos + 0.001 * normalize(lightDir); // Slightly offset to avoid self-intersection
    vec3 sphereCenter = vec3(0.0, 1.0, 0.0); // Example sphere center
    float sphereRadius = 0.5; // Example sphere radius

    if (intersectSphere(rayOrigin, lightDir, sphereCenter, sphereRadius)) {
        return true; // Point is in shadow
    }

    return false; // Point is illuminated
}

// HDR tone mapping function
vec3 toneMap(vec3 color, float exposure) {
    return 1.0 - exp(-color * exposure);
}

void main() {
    vec4 attuned;
    if (renderMode == RENDER_MODE_WIREFRAME) {
        attuned = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (renderMode == RENDER_MODE_POINTCLOUD) {
        float distance = length(fragPos - viewPos);
        float attenuation = pow(1.0 / distance, 16.0);
        vec3 color = vec3(1.0, 1.0, 1.0);
        attuned = vec4(color, attenuation);
    } else {
        attuned = vec4(0.8, 0.8, 0.8, 1.0);
    }

    vec3 ambient = 0.7 * attuned.xyz * AMBIENT_COLOR;
    vec3 lightDir = normalize(lightPos - fragPos);
    
    // Check if the point is in shadow
    #ifdef ENABLE_SHADOWS
    if (isInShadow(fragPos, lightDir)) {
        fragColor = vec4(ambient, 1.0); // Point is in shadow
        return;
    }
    #endif

    float diff = max(dot(normalize(fragPos), lightDir), 0.0);
    vec3 diffuse = diff * attuned.xyz * DIFFUSE_COLOR;
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, normalize(fragPos));
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * SPECULAR_COLOR;

    // Combine ambient, diffuse, and specular components
    vec3 finalColor = ambient + diffuse + specular;

    // Apply HDR tone mapping
    #ifdef ENABLE_HDR
    finalColor = toneMap(finalColor, HDR_EXPOSURE);
    #endif

    // Bloom effect
    #ifdef ENABLE_BLOOM
    vec3 brightColor = finalColor - ambient;
    finalColor += BLOOM_INTENSITY * brightColor;
    #endif

    fragColor = vec4(finalColor, 1.0);
}

)";
public:
	Scene();
	virtual ~Scene();
	void reset();
	bool load(const std::vector<Point3f>& points);
	bool load(const std::string& filename);
	void render(const cv::Rect& viewport, const cv::Matx44f& projection, const cv::Matx44f& view, const cv::Matx44f& modelView);
	cv::Mat_<float> pointCloudAsMat();
	std::vector<cv::Point3f> pointCloudAsVector();

	float autoScale() {
		return autoScale_;
	}

	cv::Vec3f autoCenter() {
		return autoCenter_;
	}

	void setMode(RenderMode mode) {
		mode_ = mode;
	}

	cv::Vec3f lightPosition() {
		return lightPos_;
	}

	void setLightPosition(cv::Vec3f pos) {
		lightPos_ = pos;
	}

};

} /* namespace gl */
} /* namespace v4d */
} /* namespace cv */

#endif /* MODULES_V4D_SRC_SCENE_HPP_ */
