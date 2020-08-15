#define GLFW_INCLUDE_VULKAN
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>

#include <Vulkan_Edu.h>
#include <cube_data.h>
#include <tiny_obj_loader_OLD.h>

//#define CUBE_MESH
#define OBJ_MESH
#define WIDTH 512
#define HEIGHT 512


//Custom States depending on what is needed
struct appState {

	//Vertex buffer
	struct vertices v;
	// Index buffer
	struct indices i;
	float* vBuffer;

	// Uniform buffer block object
	struct {
		VkDeviceMemory memory;
		VkBuffer buffer;
		VkDescriptorBufferInfo descriptor;
	}  uniformBufferVS;

	struct {
		glm::mat4 projectionMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	} uboVS;

	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

	std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};
	VkVertexInputBindingDescription vertexInputBinding{};
	std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributs;
};

glm::vec3 rotation = glm::vec3();
glm::vec3 cameraPos = glm::vec3();
glm::vec2 mousePos;

bool update = false;
float eyex, eyey, eyez;	// current user position

double theta, phi;		// user's position  on a sphere centered on the object
double r;
int triangles;			// number of triangles

void buildCommandBuffers(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;

	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBufInfo.pNext = nullptr;

	// Set clear values for all framebuffer attachments with loadOp set to clear
	// We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
	VkClearValue clearValues[2];
	createClearColor(context, clearValues);

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	createRenderPassCreateInfo(context, renderPassBeginInfo);
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = clearValues;

	for (int32_t i = 0; i < context.cmdBuffer.size(); ++i) {
		// Set target frame buffer
		renderPassBeginInfo.framebuffer = context.frameBuffers[i];

		res = (vkBeginCommandBuffer(context.cmdBuffer[i], &cmdBufInfo));
		assert(res == VK_SUCCESS);

		vkCmdBeginRenderPass(context.cmdBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Update dynamic viewport state
		VkViewport viewport = {};
		createViewports(context, context.cmdBuffer[i], viewport);

		// Update dynamic scissor state
		VkRect2D scissor = {};
		createScisscor(context, context.cmdBuffer[i], scissor);

		vkCmdBindDescriptorSets(context.cmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipelineLayout, 0, 1, &state.descriptorSet, 0, nullptr);
		vkCmdBindPipeline(context.cmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipeline);

		VkDeviceSize offsets[1] = { 0 };
#ifdef CUBE_MESH
		vkCmdBindVertexBuffers(context.cmdBuffer[i], 0, 1, &state.v.buffer, offsets);
		vkCmdBindIndexBuffer(context.cmdBuffer[i], state.i.buffer, 0, VK_INDEX_TYPE_UINT16);
		vkCmdDrawIndexed(context.cmdBuffer[i], 36, 1, 0, 0, 0);
#else
		vkCmdBindVertexBuffers(context.cmdBuffer[i], 0, 1, &state.v.buffer, offsets);
		vkCmdBindIndexBuffer(context.cmdBuffer[i], state.i.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(context.cmdBuffer[i], state.i.count, 1, 0, 0, 0);

#endif
		vkCmdEndRenderPass(context.cmdBuffer[i]);

		res = (vkEndCommandBuffer(context.cmdBuffer[i]));
		assert(res == VK_SUCCESS);
	}
}

void setupDescriptorPool(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;

	// We need to tell the API the number of max. requested descriptors per type
	VkDescriptorPoolSize typeCounts[1];
	// This example only uses one descriptor type (uniform buffer) and only requests one descriptor of this type
	typeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	typeCounts[0].descriptorCount = 1;

	// Create the global descriptor pool
	// All descriptors used in this example are allocated from this pool
	VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.pNext = nullptr;
	descriptorPoolInfo.poolSizeCount = 1;
	descriptorPoolInfo.pPoolSizes = typeCounts;
	// Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
	descriptorPoolInfo.maxSets = 1;

	res = (vkCreateDescriptorPool(context.device, &descriptorPoolInfo, nullptr, &context.descriptorPool));
	assert(res == VK_SUCCESS);
}

void setupDescriptorSetLayout(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;

	// Setup layout of descriptors used in this example
	// Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
	// So every shader binding should map to one descriptor set layout binding

	// Binding 0: Uniform buffer (Vertex shader)
	VkDescriptorSetLayoutBinding layoutBinding = {};
	layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layoutBinding.descriptorCount = 1;
	layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	layoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
	descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorLayout.pNext = nullptr;
	descriptorLayout.bindingCount = 1;
	descriptorLayout.pBindings = &layoutBinding;

	res = (vkCreateDescriptorSetLayout(context.device, &descriptorLayout, nullptr, &state.descriptorSetLayout));
	assert(res == VK_SUCCESS);

	// Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
	// In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
	VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
	pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pPipelineLayoutCreateInfo.pNext = nullptr;
	pPipelineLayoutCreateInfo.setLayoutCount = 1;
	pPipelineLayoutCreateInfo.pSetLayouts = &state.descriptorSetLayout;

	res = (vkCreatePipelineLayout(context.device, &pPipelineLayoutCreateInfo, nullptr, &state.pipelineLayout));
	assert(res == VK_SUCCESS);
}

void setupDescriptorSet(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;

	// Allocate a new descriptor set from the global descriptor pool
	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = context.descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &state.descriptorSetLayout;

	res = (vkAllocateDescriptorSets(context.device, &allocInfo, &state.descriptorSet));
	assert(res == VK_SUCCESS);

	VkWriteDescriptorSet writeDescriptorSet = {};

	// Binding 0 : Uniform buffer
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.dstSet = state.descriptorSet;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSet.pBufferInfo = &state.uniformBufferVS.descriptor;
	// Binds this uniform buffer to binding point 0
	writeDescriptorSet.dstBinding = 0;

	vkUpdateDescriptorSets(context.device, 1, &writeDescriptorSet, 0, nullptr);
}

void preparePipelines(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;

	VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	// The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
	pipelineCreateInfo.layout = state.pipelineLayout;
	// Renderpass this pipeline is attached to
	pipelineCreateInfo.renderPass = context.render_pass;

	// Construct the differnent states making up the pipeline

	// Input assembly state describes how primitives are assembled
	// This pipeline will assemble vertex data as a triangle lists
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
	inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	// Rasterization state
	VkPipelineRasterizationStateCreateInfo rasterizationState = {};
	rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationState.cullMode = VK_CULL_MODE_NONE;
	rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.depthClampEnable = VK_FALSE;
	rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	rasterizationState.depthBiasEnable = VK_FALSE;
	rasterizationState.lineWidth = 1.0f;

	// Color blend state describes how blend factors are calculated (if used)
	// We need one blend attachment state per color attachment (even if blending is not used
	VkPipelineColorBlendAttachmentState blendAttachmentState[1] = {};
	blendAttachmentState[0].colorWriteMask = 0xf;
	blendAttachmentState[0].blendEnable = VK_FALSE;
	VkPipelineColorBlendStateCreateInfo colorBlendState = {};
	colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendState.attachmentCount = 1;
	colorBlendState.pAttachments = blendAttachmentState;

	// Viewport state sets the number of viewports and scissor used in this pipeline
	// Note: This is actually overriden by the dynamic states (see below)
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	// Enable dynamic states
	// Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
	// To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
	// For this example we will set the viewport and scissor using dynamic states
	std::vector<VkDynamicState> dynamicStateEnables;
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
	VkPipelineDynamicStateCreateInfo dynamicState = {};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.pDynamicStates = dynamicStateEnables.data();
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

	// Depth and stencil state containing depth and stencil compare and test operations
	// We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
	VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
	depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilState.depthTestEnable = VK_TRUE;
	depthStencilState.depthWriteEnable = VK_TRUE;
	depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencilState.depthBoundsTestEnable = VK_FALSE;
	depthStencilState.back.failOp = VK_STENCIL_OP_KEEP;
	depthStencilState.back.passOp = VK_STENCIL_OP_KEEP;
	depthStencilState.back.compareOp = VK_COMPARE_OP_ALWAYS;
	depthStencilState.stencilTestEnable = VK_FALSE;
	depthStencilState.front = depthStencilState.back;

	// Multi sampling state
	// This example does not make use fo multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
	VkPipelineMultisampleStateCreateInfo multisampleState = {};
	multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampleState.pSampleMask = nullptr;


	// Vertex input state used for pipeline creation
	VkPipelineVertexInputStateCreateInfo vertexInputState = {};
	vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputState.vertexBindingDescriptionCount = 1;
	vertexInputState.pVertexBindingDescriptions = &state.vertexInputBinding;
	vertexInputState.vertexAttributeDescriptionCount = 2;
	vertexInputState.pVertexAttributeDescriptions = state.vertexInputAttributs.data();

	// Set pipeline shader stage info
	pipelineCreateInfo.stageCount = static_cast<uint32_t>(state.shaderStages.size());
	pipelineCreateInfo.pStages = state.shaderStages.data();

	// Assign the pipeline states to the pipeline creation info structure
	pipelineCreateInfo.pVertexInputState = &vertexInputState;
	pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
	pipelineCreateInfo.pRasterizationState = &rasterizationState;
	pipelineCreateInfo.pColorBlendState = &colorBlendState;
	pipelineCreateInfo.pMultisampleState = &multisampleState;
	pipelineCreateInfo.pViewportState = &viewportState;
	pipelineCreateInfo.pDepthStencilState = &depthStencilState;
	pipelineCreateInfo.renderPass = context.render_pass;
	pipelineCreateInfo.pDynamicState = &dynamicState;

	// Create rendering pipeline using the specified states
	res = (vkCreateGraphicsPipelines(context.device, context.pipelineCache, 1, &pipelineCreateInfo, nullptr, &state.pipeline));
	assert(res == VK_SUCCESS);
	// Shader modules are no longer needed once the graphics pipeline has been created
	vkDestroyShaderModule(context.device, state.shaderStages[0].module, nullptr);
	vkDestroyShaderModule(context.device, state.shaderStages[1].module, nullptr);
}

void updateUniformBuffers(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;
	// Update matrices
	state.uboVS.projectionMatrix = glm::perspective(glm::radians(60.0f), (float)context.width / (float)context.height, 0.1f, 256.0f);

	state.uboVS.viewMatrix = glm::lookAt(glm::vec3(eyex, eyey, eyez),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f));

	state.uboVS.modelMatrix = glm::mat4(1.0f);
	state.uboVS.modelMatrix = glm::rotate(state.uboVS.modelMatrix, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	state.uboVS.modelMatrix = glm::rotate(state.uboVS.modelMatrix, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	state.uboVS.modelMatrix = glm::rotate(state.uboVS.modelMatrix, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

	// Map uniform buffer and update it
	uint8_t* pData;
	res = (vkMapMemory(context.device, state.uniformBufferVS.memory, 0, sizeof(state.uboVS), 0, (void**)&pData));
	memcpy(pData, &state.uboVS, sizeof(state.uboVS));
	// Unmap after data has been copied
	// Note: Since we requested a host coherent memory type for the uniform buffer, the write is instantly visible to the GPU
	vkUnmapMemory(context.device, state.uniformBufferVS.memory);
}

void prepareUniformBuffers(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;
	// Prepare and initialize a uniform buffer block containing shader uniforms
	// Single uniforms like in OpenGL are no longer present in Vulkan. All Shader uniforms are passed via uniform buffer blocks
	VkMemoryRequirements memReqs;

	// Vertex shader uniform buffer block
	VkBufferCreateInfo bufferInfo = {};
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.pNext = nullptr;
	allocInfo.allocationSize = 0;
	allocInfo.memoryTypeIndex = 0;

	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(state.uboVS);
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

	bindBufferToMem(context, bufferInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		state.uniformBufferVS.buffer, state.uniformBufferVS.memory);

	// Store information in the uniform's descriptor that is used by the descriptor set
	state.uniformBufferVS.descriptor.buffer = state.uniformBufferVS.buffer;
	state.uniformBufferVS.descriptor.offset = 0;
	state.uniformBufferVS.descriptor.range = sizeof(state.uboVS);

	updateUniformBuffers(context, state);
}

#ifdef CUBE_MESH
void prepareVertices(struct LHContext& context, struct appState& state, bool useStagingBuffers) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;

	VkBufferCreateInfo buf_info = {};
	VkMemoryRequirements mem_reqs;
	VkMemoryAllocateInfo alloc_info = {};
	uint8_t* pData;

	int i;
	int k;
	float vertices[8][4] = {
		{ -1.0, -1.0, -1.0, 1.0 }, //0
		{ -1.0, -1.0, 1.0, 1.0 }, //1
		{ -1.0, 1.0, -1.0, 1.0 }, //2
		{ -1.0, 1.0, 1.0, 1.0 }, //3
		{ 1.0, -1.0, -1.0, 1.0 }, //4
		{ 1.0, -1.0, 1.0, 1.0 }, //5
		{ 1.0, 1.0, -1.0, 1.0 }, //6
		{ 1.0, 1.0, 1.0, 1.0 } //7
	};

	float normals[8][3] = {
		{ -1.0, -1.0, -1.0 }, //0
		{ -1.0, -1.0, 1.0 }, //1
		{ -1.0, 1.0, -1.0 }, //2
		{ -1.0, 1.0, 1.0 }, //3
		{ 1.0, -1.0, -1.0 }, //4
		{ 1.0, -1.0, 1.0 }, //5
		{ 1.0, 1.0, -1.0 }, //6
		{ 1.0, 1.0, 1.0 } //7
	};

	uint16_t indices[36] = { 3, 1, 0, 0, 2, 3,
		0, 4, 5, 5, 1, 0,
		2, 6, 7, 7, 3, 2,
		0, 4, 6, 0, 2, 6,
		1, 5, 7, 7, 3, 1,
		7, 5, 4, 4, 6, 7 };

	state.vBuffer = new float[7 * 8];
	k = 0;
	for (i = 0; i < 8; i++) {
		state.vBuffer[k++] = vertices[i][0];
		state.vBuffer[k++] = vertices[i][1];
		state.vBuffer[k++] = vertices[i][2];
		state.vBuffer[k++] = vertices[i][3];
		state.vBuffer[k++] = normals[i][0];
		state.vBuffer[k++] = normals[i][1];
		state.vBuffer[k++] = normals[i][2];
	}

	uint32_t dataSize = 7 * 8 * sizeof(float);
	uint32_t dataStride = 7 * sizeof(float);

	//mapIndiciesToGPU(context, indices, sizeof(indices), state.i.buffer, state.i.memory);
	//mapVerticiesToGPU(context, state.vBuffer, dataSize, state.v.buffer, state.v.memory);

	buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buf_info.pNext = NULL;
	buf_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	buf_info.size = sizeof(indices);
	buf_info.queueFamilyIndexCount = 0;
	buf_info.pQueueFamilyIndices = NULL;
	buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	buf_info.flags = 0;
	res = vkCreateBuffer(context.device, &buf_info, NULL, &state.i.buffer);
	assert(res == VK_SUCCESS);
	vkGetBufferMemoryRequirements(context.device, state.i.buffer, &mem_reqs);
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.pNext = NULL;
	alloc_info.memoryTypeIndex = 0;
	alloc_info.allocationSize = mem_reqs.size;
	pass = memory_type_from_properties(context, mem_reqs.memoryTypeBits,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&alloc_info.memoryTypeIndex);
	assert(pass && "No mappable coherent memory");
	VkDeviceMemory mem;
	res = vkAllocateMemory(context.device, &alloc_info, NULL, &mem);
	assert(res == VK_SUCCESS);
	res = vkMapMemory(context.device, mem, 0, mem_reqs.size, 0, (void**)
		&pData);
	assert(res == VK_SUCCESS);
	memcpy(pData, indices, sizeof(indices));
	vkUnmapMemory(context.device, mem);
	res = vkBindBufferMemory(context.device, state.i.buffer, mem, 0);
	assert(res == VK_SUCCESS);

	buf_info = {};
	buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buf_info.pNext = NULL;
	buf_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	buf_info.size = dataSize;
	buf_info.queueFamilyIndexCount = 0;
	buf_info.pQueueFamilyIndices = NULL;
	buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	buf_info.flags = 0;
	res = vkCreateBuffer(context.device, &buf_info, NULL, &state.v.buffer);
	assert(res == VK_SUCCESS);

	mem_reqs = {};
	vkGetBufferMemoryRequirements(context.device, state.v.buffer,
		&mem_reqs);

	alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.pNext = NULL;
	alloc_info.memoryTypeIndex = 0;

	alloc_info.allocationSize = mem_reqs.size;
	pass = memory_type_from_properties(context, mem_reqs.memoryTypeBits,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&alloc_info.memoryTypeIndex);
	assert(pass && "No mappable, coherent memory");

	res = vkAllocateMemory(context.device, &alloc_info, NULL,
		&(state.v.memory));
	assert(res == VK_SUCCESS);
	state.v.buffer_info.range = mem_reqs.size;
	state.v.buffer_info.offset = 0;

	res = vkMapMemory(context.device, state.v.memory, 0, mem_reqs.size, 0,
		(void**)&pData);
	assert(res == VK_SUCCESS);

	memcpy(pData, state.vBuffer, dataSize);

	vkUnmapMemory(context.device, state.v.memory);

	res = vkBindBufferMemory(context.device, state.v.buffer,
		state.v.memory, 0);
	assert(res == VK_SUCCESS);

	//// Vertex input descriptions 
	//// Specifies the vertex input parameters for a pipeline

	//// Vertex input binding
	//// This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)

	state.vertexInputBinding.binding = 0;
	state.vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	state.vertexInputBinding.stride = dataStride;


	// Inpute attribute bindings describe shader attribute locations and memory layouts
	// These match the following shader layout
	//	layout (location = 0) in vec3 inPos;
	//	layout (location = 1) in vec3 inNormal;
	// Attribute location 0: Position
	//// Attribute location 1: Normal
	state.vertexInputAttributs[0].binding = 0;
	state.vertexInputAttributs[0].location = 0;
	state.vertexInputAttributs[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	state.vertexInputAttributs[0].offset = 0;
	state.vertexInputAttributs[1].binding = 0;
	state.vertexInputAttributs[1].location = 1;
	state.vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.vertexInputAttributs[1].offset = 16;

}
#endif

#ifdef OBJ_MESH
void prepareVertices(struct LHContext& context, struct appState& state, bool useStagingBuffers) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;

	float* vertices;
	float* normals;
	uint32_t* indices;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	int nv;
	int nn;
	int ni;
	int i;

	VkBufferCreateInfo buf_info = {};
	VkMemoryRequirements mem_reqs;
	VkMemoryAllocateInfo alloc_info = {};
	uint8_t* pData;

	std::string err = tinyobj::LoadObj(shapes, materials, "data/sphere.obj", 0);

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return;
	}

	/*  Retrieve the vertex coordinate data */

	nv = (int)shapes[0].mesh.positions.size();
	vertices = new GLfloat[nv];
	for (i = 0; i < nv; i++) {
		vertices[i] = shapes[0].mesh.positions[i];
	}

	/*  Retrieve the vertex normals */

	nn = (int)shapes[0].mesh.normals.size();
	normals = new GLfloat[nn];
	for (i = 0; i < nn; i++) {
		normals[i] = shapes[0].mesh.normals[i];
	}

	/*  Retrieve the triangle indices */

	ni = (int)shapes[0].mesh.indices.size();
	triangles = ni / 3;
	indices = new uint32_t[ni];
	for (i = 0; i < ni; i++) {
		indices[i] = shapes[0].mesh.indices[i];
	}


	state.vBuffer = new float[nv + nn];
	int k = 0;
	for (i = 0; i < nv / 3; i++) {
		state.vBuffer[k++] = vertices[3 * i];
		state.vBuffer[k++] = vertices[3 * i + 1];
		state.vBuffer[k++] = vertices[3 * i + 2];
		state.vBuffer[k++] = normals[3 * i];
		state.vBuffer[k++] = normals[3 * i + 1];
		state.vBuffer[k++] = normals[3 * i + 2];
	}

	uint32_t dataSize = (nv + nn) * sizeof(state.vBuffer[0]);
	uint32_t dataStride = 6 * (sizeof( float));

	state.i.count = ni;

	mapIndiciesToGPU(context, indices, sizeof(indices[0]) * ni, state.i.buffer, state.i.memory);
	mapVerticiesToGPU(context, state.vBuffer, dataSize, state.v.buffer, state.v.memory);

	//// Vertex input descriptions 
	//// Specifies the vertex input parameters for a pipeline

	//// Vertex input binding
	//// This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)

	state.vertexInputBinding.binding = 0;
	state.vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	state.vertexInputBinding.stride = dataStride;


	// Inpute attribute bindings describe shader attribute locations and memory layouts
	// These match the following shader layout
	//	layout (location = 0) in vec3 inPos;
	//	layout (location = 1) in vec3 inNormal;
	// Attribute location 0: Position
	//// Attribute location 1: Normal
	state.vertexInputAttributs[0].binding = 0;
	state.vertexInputAttributs[0].location = 0;
	state.vertexInputAttributs[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	state.vertexInputAttributs[0].offset = 0;
	state.vertexInputAttributs[1].binding = 0;
	state.vertexInputAttributs[1].location = 1;
	state.vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.vertexInputAttributs[1].offset = sizeof(float);

}
#endif // OBJ_MESH

void renderLoop(struct LHContext& context, struct appState& state) {

	while (!glfwWindowShouldClose(context.window)) {
		glfwPollEvents();
		draw(context);
		if (update) {
			updateUniformBuffers(context, state);
			update = false;
		}
	}

	// Flush device to make sure all resources can be freed
	if (context.device != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(context.device);
	}
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	if (key == GLFW_KEY_A && action == GLFW_PRESS) {
		phi -= 0.1;
		update = true;
	}
	if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		phi += 0.1;
		update = true;
	}
	if (key == GLFW_KEY_W && action == GLFW_PRESS) {
		theta += 0.1;
		update = true;
	}
	if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		theta -= 0.1;
		update = true;
	}

	eyex = (float)(r * sin(theta) * cos(phi));
	eyey = (float)(r * sin(theta) * sin(phi));
	eyez = (float)(r * cos(theta));

}

void prepareShaders(struct LHContext& context, struct appState& state) {
	// Shaders

	// Vertex shader
	createShaderStage(context, "./shaders/shader.vert", VK_SHADER_STAGE_VERTEX_BIT, state.shaderStages[0]);
	assert(state.shaderStages[0].module != VK_NULL_HANDLE);

	// Fragment shader
	createShaderStage(context, "./shaders/shader.frag", VK_SHADER_STAGE_FRAGMENT_BIT, state.shaderStages[1]);
	assert(state.shaderStages[1].module != VK_NULL_HANDLE);
}

int main() {
	eyex = 0.0;
	eyez = 2.0;
	eyey = 7.0;

	theta = 1.5;
	phi = 1.5;
	r = 10.0;


	struct LHContext context = {};
	struct appState state = {};
	createInstance(context);
	createDeviceInfo(context);
	createWindowContext(context, 1280, 720);
	createSwapChainExtention(context);
	createDevice(context);
	createDeviceQueue(context);
	createSynchObject(context);
	createCommandPool(context);
	createSwapChain(context);
	createCommandBuffer(context);
	createSynchPrimitive(context);
	createDepthBuffers(context);
	createRenderPass(context);
	createPipeLineCache(context);
	createFrameBuffer(context);
	prepareSynchronizationPrimitives(context);

	//---> Implement our own functions
	prepareVertices(context, state, false);
	prepareUniformBuffers(context, state);
	setupDescriptorSetLayout(context, state);
	prepareShaders(context, state);
	preparePipelines(context, state);
	setupDescriptorPool(context, state);
	setupDescriptorSet(context, state);
	buildCommandBuffers(context, state);

	glfwSetKeyCallback(context.window, key_callback);

	renderLoop(context, state);

	return 0;
}