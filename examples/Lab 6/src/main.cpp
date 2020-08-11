#define GLFW_INCLUDE_VULKAN
#define LHTexture
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>

#include <Vulkan_Edu.h>
#include <tiny_obj_loader.h>
#include <texture.h>
#include <FreeImage.h>

#define OBJ_MESH
#define WIDTH 512
#define HEIGHT 512

struct MipMap {
	VkExtent2D					dimensions_size;
	uint32_t					byte_size;
	FIBITMAP* image;
	uint32_t					offset;
};

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float uv[2];
	float normal[3];
};

uint32_t indexCount;

//Custom States depending on what is needed
struct appState {
	struct Model {
		struct Matricies {
			glm::mat4 projectionMatrix;
			glm::mat4 viewMatrix;
			glm::mat4 modelMatrix;
		}uniformVS;

		struct Lighting {
			glm::vec3 lightPos;
			float ambientStrenght;
			float specularStrenght;
		}uniformFS;

		struct vertices v;
		struct indices i;

		//Index 1: VS (Matricies)
		//Index 2: FS (Lighting)
		struct UniformBufferBlock {
			VkDeviceMemory memory;
			VkBuffer buffer;
			VkDescriptorBufferInfo descriptor;
		}uniformBuffer[2];

		struct Texture {
			VkSampler sampler;
			VkImage image;
			VkImageLayout imageLayout;
			VkDeviceMemory deviceMemory;
			VkImageView view;
			VkBuffer buffer;
			uint32_t width, height, size, depth;
			uint32_t mipLevels;
			void* data;
			VkDescriptorImageInfo descriptor;
		} textureData;

		VkDescriptorSet descriptorSet;
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};
		std::array<VkVertexInputAttributeDescription, 3> vertexInputAttributs;
		VkVertexInputBindingDescription vertexInputBinding{};

		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		VkDescriptorSetLayout descriptorSetLayout;
	};
	Model mesh = {};
};

glm::vec3 rotation = glm::vec3();
glm::vec3 cameraPos = glm::vec3();
glm::vec2 mousePos;

bool update = false;
float eyex, eyey, eyez;	// current user position

double theta, phi;		// user's position  on a sphere centered on the object
double r;
int triangles;			// number of triangles

//
///*
//	Upload texture image data to the GPU
//
//	Vulkan offers two types of image tiling (memory layout):
//
//	Linear tiled images:
//		These are stored as is and can be copied directly to. But due to the linear nature they're not a good match for GPUs and format and feature support is very limited.
//		It's not advised to use linear tiled images for anything else than copying from host to GPU if buffer copies are not an option.
//		Linear tiling is thus only implemented for learning purposes, one should always prefer optimal tiled image.
//
//	Optimal tiled images:
//		These are stored in an implementation specific layout matching the capability of the hardware. They usually support more formats and features and are much faster.
//		Optimal tiled images are stored on the device and not accessible by the host. So they can't be written directly to (like liner tiled images) and always require
//		some sort of data copy, either from a buffer or	a linear tiled image.
//
//	In Short: Always use optimal tiled images for rendering.
//*/
//

//
//void prepareTexture(struct LHContext& context, struct appState& state, std::string filename, int index) {
//	VkResult U_ASSERT_ONLY res;
//	bool U_ASSERT_ONLY pass;
//
//	VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
//
//	//Grab the File Format
//	FREE_IMAGE_FORMAT formatT = FreeImage_GetFIFFromFilename(filename.c_str());
//	std::wstring filenameS(filename.begin(), filename.end());
//	FIBITMAP* bitImage = FreeImage_LoadU(formatT, filenameS.c_str());
//
//	//Checks if the format exist within FreeImage
//	if (format == FIF_UNKNOWN) {
//		printf("Unknown file type for texture image file %s\n", filename);
//		return;
//	}
//
//	//Checks if the image is 32 Bit (RGBA)
//	if (FreeImage_GetBPP(bitImage) != 32) {
//		FIBITMAP* bitImageTemp = FreeImage_ConvertTo32Bits(bitImage);
//		FreeImage_Unload(bitImage);
//		bitImage = bitImageTemp;
//	}
//
//	FreeImage_FlipVertical(bitImage);
//	FREE_IMAGE_TYPE bitImageType = FreeImage_GetImageType(bitImage);
//	FREE_IMAGE_COLOR_TYPE  btImageColorType = FreeImage_GetColorType(bitImage);
//
//	state.text[index].width = FreeImage_GetWidth(bitImage);
//	state.text[index].height = FreeImage_GetHeight(bitImage);
//	state.text[index].depth = FreeImage_GetBPP(bitImage);
//	state.text[index].size = state.text[index].width * state.text[index].height * (state.text[index].depth / 8);
//	state.text[index].data = FreeImage_GetBits(bitImage);
//
//	std::vector<MipMap> mipmaps;
//	VkExtent2D sizeExtents = { 0, 0 };
//	mipmaps.reserve(16);
//	{
//		uint32_t currentOffsets = 0;
//		MipMap last;
//		last.dimensions_size = sizeExtents;
//		last.byte_size = state.text[index].size;
//		last.image = bitImage;
//		last.offset = currentOffsets;
//		mipmaps.push_back(last);
//
//		while (last.dimensions_size.width != 1 && last.dimensions_size.height != 1) {
//			VkExtent2D current_dim_size = { last.dimensions_size.width / 2, last.dimensions_size.height / 2 };
//			if (current_dim_size.width < 1)	current_dim_size.width = 1;
//			if (current_dim_size.height < 1)	current_dim_size.height = 1;
//			uint32_t current_byte_size = current_dim_size.width * current_dim_size.height * (state.text[index].depth / 8);
//			currentOffsets += ((last.byte_size / 4 + !!(last.byte_size % 4)) * 4);
//
//			MipMap current;
//			current.dimensions_size = current_dim_size;
//			current.byte_size = current_byte_size;
//			current.image = FreeImage_Rescale(last.image, current_dim_size.width, current_dim_size.height);
//			current.offset = currentOffsets;
//
//			mipmaps.push_back(current);
//			last = current;
//		}
//
//	}
//
//	uint32_t request_buffer_size = mipmaps.back().offset + mipmaps.back().byte_size;
//
//	VkBuffer stagging_buffer = VK_NULL_HANDLE;
//	VkDeviceMemory stagging_buffer_memory = VK_NULL_HANDLE;
//
//	{
//		VkBufferCreateInfo bufferCreateInfo = {};
//		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
//		bufferCreateInfo.flags = 0;
//		bufferCreateInfo.size = request_buffer_size;
//		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
//		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//		res = vkCreateBuffer(context.device, &bufferCreateInfo, nullptr, &stagging_buffer);
//		assert(res == VK_SUCCESS);
//
//		VkMemoryRequirements memory_requirements{};
//		VkMemoryAllocateInfo allocInfo = {};
//		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
//		allocInfo.pNext = nullptr;
//		allocInfo.allocationSize = 0;
//		allocInfo.memoryTypeIndex = 0;
//
//
//		vkGetBufferMemoryRequirements(context.device, stagging_buffer, &memory_requirements);
//		allocInfo.allocationSize = memory_requirements.size;
//		pass = memory_type_from_properties(context, memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &allocInfo.memoryTypeIndex);
//		assert(pass && "No mappable coherent memory");
//		res = (vkAllocateMemory(context.device, &allocInfo, nullptr, &stagging_buffer_memory));
//		assert(res == VK_SUCCESS);
//		res = (vkBindBufferMemory(context.device, stagging_buffer, stagging_buffer_memory, 0));
//		assert(res == VK_SUCCESS);
//
//		{
//			uint8_t *data;
//			res = vkMapMemory(context.device, stagging_buffer_memory, 0, memory_requirements.size, 0,(void**)&data);
//			assert(res == VK_SUCCESS);
//			for (MipMap &mip: mipmaps) {
//				uint32_t mipByteSize = mip.dimensions_size.width * mip.dimensions_size.height * (state.text[index].depth / 8);
//				std::memcpy(&data[mip.offset], FreeImage_GetBits(mip.image), mipByteSize);
//			}
//			vkUnmapMemory(context.device, stagging_buffer_memory);
//		}
//	}
//
//	for (MipMap& mip : mipmaps) {
//		FreeImage_Unload(mip.image);
//		mip.image = nullptr;
//	}
//
//	VkImage mappableImage;
//	VkDeviceMemory mappableMemory;
//
//	VkMemoryRequirements memReqs;
//	VkMemoryAllocateInfo allocInfo = {};
//	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
//	allocInfo.pNext = nullptr;
//	allocInfo.allocationSize = 0;
//	allocInfo.memoryTypeIndex = 0;
//
//	{
//		VkImageCreateInfo imageCreateInfo = {};
//		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
//		imageCreateInfo.flags = 0;
//		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
//		imageCreateInfo.format = format;
//		imageCreateInfo.extent = { state.text[index].width, state.text[index].height, 1 };
//		imageCreateInfo.mipLevels = uint32_t(mipmaps.size());
//		imageCreateInfo.arrayLayers = 1;
//		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
//		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
//		imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
//		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
//		res = vkCreateImage(context.device, &imageCreateInfo, nullptr,&mappableImage);
//		assert(res == VK_SUCCESS);
//
//		vkGetImageMemoryRequirements(context.device, mappableImage, &memReqs);
//		allocInfo.allocationSize = memReqs.size;
//		pass = memory_type_from_properties(context, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &allocInfo.memoryTypeIndex);
//		assert(pass && "No mappable coherent memory");
//		res = (vkAllocateMemory(context.device, &allocInfo, nullptr, &mappableMemory));
//		assert(res == VK_SUCCESS);
//		res = (vkBindImageMemory(context.device, mappableImage, mappableMemory, 0));
//		assert(res == VK_SUCCESS);
//
//		state.text[index].image = mappableImage;
//		state.text[index].deviceMemory = mappableMemory;
//
//		VkImageViewCreateInfo image_view_create_info{};
//		image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
//		image_view_create_info.flags = 0;
//		image_view_create_info.image = mappableImage;
//		image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
//		image_view_create_info.format = format;
//		image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
//		image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
//		image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
//		image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
//		image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
//		image_view_create_info.subresourceRange.baseMipLevel = 0;
//		image_view_create_info.subresourceRange.levelCount = uint32_t(mipmaps.size());
//		image_view_create_info.subresourceRange.baseArrayLayer = 0;
//		image_view_create_info.subresourceRange.layerCount = 1;
//		res = vkCreateImageView(context.device, &image_view_create_info, nullptr, &state.text[index].view);
//		assert(res == VK_SUCCESS);
//	}
//
//	{
//
//		VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
//		VkCommandBufferAllocateInfo buffer_allocate_info{};
//		buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
//		buffer_allocate_info.commandPool = context.cmd_pool;
//		buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
//		buffer_allocate_info.commandBufferCount = 1;
//		res = vkAllocateCommandBuffers(context.device, &buffer_allocate_info, &cmdBuffer);
//		assert(res == VK_SUCCESS);
//
//		VkCommandBufferBeginInfo buffer_begin_info{};
//		buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//		buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
//		res = vkBeginCommandBuffer(cmdBuffer, &buffer_begin_info);
//		assert(res == VK_SUCCESS);
//
//		VkImageMemoryBarrier image_memory_barrier = {};
//		image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
//		image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
//		image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
//		image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
//		image_memory_barrier.subresourceRange.baseMipLevel = 0;
//		image_memory_barrier.subresourceRange.levelCount = uint32_t(mipmaps.size());
//		image_memory_barrier.subresourceRange.baseArrayLayer = 0;
//		image_memory_barrier.subresourceRange.layerCount = 1;
//
//		// translate the layout of the final image from undefined to transfer destination optimal
//		image_memory_barrier.srcAccessMask = 0;
//		image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
//		image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
//		image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
//		image_memory_barrier.image = mappableImage;
//
//		vkCmdPipelineBarrier(cmdBuffer,
//			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
//			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
//			0,
//			0, nullptr,
//			0, nullptr,
//			1, &image_memory_barrier);
//
//		// copy buffer to image
//		std::vector<VkBufferImageCopy> regions;
//		regions.reserve(16);
//		for (int i = 0; i < mipmaps.size(); ++i) {
//			auto& m = mipmaps[i];
//			VkBufferImageCopy region{};
//			region.bufferOffset = m.offset;
//			region.bufferRowLength = 0;
//			region.bufferImageHeight = 0;
//			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
//			region.imageSubresource.mipLevel = i;
//			region.imageSubresource.baseArrayLayer = 0;
//			region.imageSubresource.layerCount = 1;
//			region.imageOffset = { 0, 0, 0 };
//			region.imageExtent = { m.dimensions_size.width, m.dimensions_size.height, 1 };
//			regions.push_back(region);
//		}
//
//		vkCmdCopyBufferToImage(cmdBuffer,
//			stagging_buffer,
//			mappableImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
//			uint32_t(regions.size()), regions.data());
//
//		image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
//		image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
//		image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
//		image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
//		image_memory_barrier.image = mappableImage;
//		vkCmdPipelineBarrier(cmdBuffer,
//			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
//			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
//			0,
//			0, nullptr,
//			0, nullptr,
//			1, &image_memory_barrier);
//
//		res = vkEndCommandBuffer(cmdBuffer);
//		assert(res == VK_SUCCESS);
//
//		VkSubmitInfo submit_info{};
//		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
//		submit_info.commandBufferCount = 1;
//		submit_info.pCommandBuffers = &cmdBuffer;
//		res = vkQueueSubmit(context.queue, 1, &submit_info, VK_NULL_HANDLE);
//		assert(res == VK_SUCCESS);
//
//		vkQueueWaitIdle(context.queue);
//		vkFreeCommandBuffers(context.device, context.cmd_pool, 1, &cmdBuffer);
//	}
//
//	vkDestroyBuffer(context.device, stagging_buffer, nullptr);
//	vkFreeMemory(context.device, stagging_buffer_memory, nullptr);
//
//	{
//			
//		// Create a texture sampler
//		// In Vulkan textures are accessed by samplers
//		// This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
//		// Note: Similar to the samplers available with OpenGL 3.3
//		VkSamplerCreateInfo sampler = {};
//		sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
//		sampler.maxAnisotropy = 1.0f;
//		sampler.magFilter = VK_FILTER_LINEAR;
//		sampler.minFilter = VK_FILTER_LINEAR;
//		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
//		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
//		sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
//		sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
//		sampler.mipLodBias = 0.0f;
//		sampler.compareOp = VK_COMPARE_OP_NEVER;
//		sampler.minLod = 0.0f;
//		// Set max level-of-detail to mip level count of the texture
//		sampler.maxLod = 0.0f;
//		sampler.maxAnisotropy = 1.0;
//		sampler.anisotropyEnable = VK_FALSE;
//		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
//		res = (vkCreateSampler(context.device, &sampler, nullptr, &state.text[index].sampler));
//		assert(res == VK_SUCCESS);
//
//		state.text[index].descriptor.sampler = state.text[index].sampler;
//		state.text[index].descriptor.imageView = state.text[index].view;
//		state.text[index].descriptor.imageLayout = state.text[index].imageLayout;
//	}
//}
//


void buildCommandBuffer(struct LHContext& context, struct appState& state) {
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

		vkCmdBindPipeline(context.cmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, state.mesh.pipeline);
		vkCmdBindDescriptorSets(context.cmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, state.mesh.pipelineLayout, 0, 1, &state.mesh.descriptorSet, 0, NULL);;

		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(context.cmdBuffer[i], 0, 1, &state.mesh.v.buffer, offsets);
		vkCmdBindIndexBuffer(context.cmdBuffer[i], state.mesh.i.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdDrawIndexed(context.cmdBuffer[i], state.mesh.i.count, 1, 0, 0, 0);
		vkCmdEndRenderPass(context.cmdBuffer[i]);

		res = (vkEndCommandBuffer(context.cmdBuffer[i]));
		assert(res == VK_SUCCESS);
	}
}

void preparePipelines(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;

	VkPipelineLayoutCreateInfo pipelineCreateInfo{};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	// The pipeline layout is based on the descriptor set layout we created above
	pipelineCreateInfo.setLayoutCount = 1;
	pipelineCreateInfo.pSetLayouts = &state.mesh.descriptorSetLayout;
	res = (vkCreatePipelineLayout(context.device, &pipelineCreateInfo, nullptr, &state.mesh.pipelineLayout));
	assert(res == VK_SUCCESS);
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
	vertexInputState.pVertexBindingDescriptions = &state.mesh.vertexInputBinding;
	vertexInputState.vertexAttributeDescriptionCount = 3;
	vertexInputState.pVertexAttributeDescriptions = state.mesh.vertexInputAttributs.data();

	VkGraphicsPipelineCreateInfo pipelineGraphicCreateInfo = {};
	pipelineGraphicCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	// The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
	pipelineGraphicCreateInfo.layout = state.mesh.pipelineLayout;
	// Renderpass this pipeline is attached to
	pipelineGraphicCreateInfo.renderPass = context.render_pass;

	// Assign the pipeline states to the pipeline creation info structure
	pipelineGraphicCreateInfo.pVertexInputState = &vertexInputState;
	pipelineGraphicCreateInfo.pInputAssemblyState = &inputAssemblyState;
	pipelineGraphicCreateInfo.pRasterizationState = &rasterizationState;
	pipelineGraphicCreateInfo.pColorBlendState = &colorBlendState;
	pipelineGraphicCreateInfo.pMultisampleState = &multisampleState;
	pipelineGraphicCreateInfo.pViewportState = &viewportState;
	pipelineGraphicCreateInfo.pDepthStencilState = &depthStencilState;
	pipelineGraphicCreateInfo.renderPass = context.render_pass;
	pipelineGraphicCreateInfo.pDynamicState = &dynamicState;


	// Set pipeline shader stage info
	pipelineGraphicCreateInfo.stageCount = static_cast<uint32_t>(state.mesh.shaderStages.size());
	pipelineGraphicCreateInfo.pStages = state.mesh.shaderStages.data();

	// Create rendering pipeline using the specified states
	res = (vkCreateGraphicsPipelines(context.device, context.pipelineCache, 1, &pipelineGraphicCreateInfo, nullptr, &state.mesh.pipeline));
	assert(res == VK_SUCCESS);
}

void prepareShader(struct LHContext& context, struct appState& state, std::string filenameVS, std::string filenameFS) {
	VkResult U_ASSERT_ONLY res;

	// Vertex shader
	createShaderStage(context, filenameVS, VK_SHADER_STAGE_VERTEX_BIT, state.mesh.shaderStages[0]);
	assert(state.mesh.shaderStages[0].module != VK_NULL_HANDLE);

	// Fragment shader
	createShaderStage(context, filenameFS, VK_SHADER_STAGE_FRAGMENT_BIT, state.mesh.shaderStages[1]);
	assert(state.mesh.shaderStages[1].module != VK_NULL_HANDLE);
}

void prepareTexture2(struct LHContext& context, struct appState& state, std::string filename) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;

	VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

	//Grab the File Format
	FREE_IMAGE_FORMAT formatT = FreeImage_GetFIFFromFilename(filename.c_str());
	std::wstring filenameS(filename.begin(), filename.end());
	FIBITMAP* bitImage = FreeImage_LoadU(formatT, filenameS.c_str());

	//Checks if the format exist within FreeImage
	if (format == FIF_UNKNOWN) {
		printf("Unknown file type for texture image file %s\n", filename);
		return;
	}

	//Checks if the image is 32 Bit (RGBA)
	if (FreeImage_GetBPP(bitImage) != 32) {
		FIBITMAP* bitImageTemp = FreeImage_ConvertTo32Bits(bitImage);
		FreeImage_Unload(bitImage);
		bitImage = bitImageTemp;
	}

	FreeImage_FlipVertical(bitImage);
	FREE_IMAGE_TYPE bitImageType = FreeImage_GetImageType(bitImage);
	FREE_IMAGE_COLOR_TYPE  btImageColorType = FreeImage_GetColorType(bitImage);

	state.mesh.textureData.width = FreeImage_GetWidth(bitImage);
	state.mesh.textureData.height = FreeImage_GetHeight(bitImage);
	state.mesh.textureData.depth = FreeImage_GetBPP(bitImage);
	state.mesh.textureData.size = state.mesh.textureData.width * state.mesh.textureData.height * (state.mesh.textureData.depth / 8);
	state.mesh.textureData.data = FreeImage_GetBits(bitImage);

	VkMemoryAllocateInfo memAllocInfo = {};
	memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	VkMemoryRequirements memReqs = {};

	//Linear tiled image
	VkImage mappableImage;
	VkDeviceMemory mappableMemory;

	VkImageCreateInfo imageCreateInfo = {};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.format = format;
	imageCreateInfo.mipLevels = 1;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
	imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
	imageCreateInfo.extent = { state.mesh.textureData.width,state.mesh.textureData.height, 1 };
	res = vkCreateImage(context.device, &imageCreateInfo, nullptr, &mappableImage);
	assert(res == VK_SUCCESS);

	vkGetImageMemoryRequirements(context.device, mappableImage, &memReqs);
	memAllocInfo.allocationSize = memReqs.size;
	pass = memory_type_from_properties(context, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &memAllocInfo.memoryTypeIndex);
	assert(pass && "No mappable coherent memory");

	res = vkAllocateMemory(context.device, &memAllocInfo, nullptr, &mappableMemory);
	assert(res == VK_SUCCESS);

	res = vkBindImageMemory(context.device, mappableImage, mappableMemory, 0);
	assert(res == VK_SUCCESS);

	void* data;
	res = vkMapMemory(context.device, mappableMemory, 0, memReqs.size, 0, &data);
	assert(res == VK_SUCCESS);
	memcpy(data, state.mesh.textureData.data, memReqs.size);
	vkUnmapMemory(context.device, mappableMemory);

	state.mesh.textureData.image = mappableImage;
	state.mesh.textureData.deviceMemory = mappableMemory;
	state.mesh.textureData.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.commandPool = context.cmd_pool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;

	VkCommandBuffer copyCmd;
	res = vkAllocateCommandBuffers(context.device, &commandBufferAllocateInfo, &copyCmd);
	assert(res == VK_SUCCESS);

	//Start to record new command buffer

	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	res = vkBeginCommandBuffer(copyCmd, &cmdBufInfo);

	// The sub resource range describes the regions of the image we will be transition
	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0;
	subresourceRange.levelCount = 1;
	subresourceRange.layerCount = 1;

	VkImageMemoryBarrier imageMemoryBarrier{};
	imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageMemoryBarrier.image = state.mesh.textureData.image;
	imageMemoryBarrier.subresourceRange = subresourceRange;
	imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
	imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
	imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// Insert a memory dependency at the proper pipeline stages that will execute the image layout transition
	// Source pipeline stage is host write/read exection (VK_PIPELINE_STAGE_HOST_BIT)
	// Destination pipeline stage fragment shader access (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
	vkCmdPipelineBarrier(
		copyCmd,
		VK_PIPELINE_STAGE_HOST_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &imageMemoryBarrier);

	if (copyCmd == VK_NULL_HANDLE) {
		return;
	}

	//Flush the command Buffer so that its not in the pool
	res = vkEndCommandBuffer(copyCmd);
	assert(res == VK_SUCCESS);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &copyCmd;

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = 0;
	VkFence fence;

	res == vkCreateFence(context.device, &fenceInfo, nullptr, &fence);
	assert(res == VK_SUCCESS);

	res == vkQueueSubmit(context.queue, 1, &submitInfo, fence);
	assert(res == VK_SUCCESS);
	res = (vkQueueWaitIdle(context.queue));
	assert(res == VK_SUCCESS);

	res = vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX);
	assert(res == VK_SUCCESS);

	vkDestroyFence(context.device, fence, nullptr);
	vkFreeCommandBuffers(context.device, context.cmd_pool, 1, &copyCmd);

	VkSamplerCreateInfo sampler = {};
	sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler.maxAnisotropy = 1.0f;
	sampler.magFilter = VK_FILTER_LINEAR;
	sampler.minFilter = VK_FILTER_LINEAR;
	sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.mipLodBias = 0.0f;
	sampler.compareOp = VK_COMPARE_OP_NEVER;
	sampler.minLod = 0.0f;
	sampler.maxLod = 0.0f;

	if (context.deviceFeatures.samplerAnisotropy) {
		sampler.maxAnisotropy = context.deviceProperties.limits.maxSamplerAnisotropy;//vulkanDevice->properties.limits.maxSamplerAnisotropy;
		sampler.anisotropyEnable = VK_TRUE;
	}
	else {
		sampler.maxAnisotropy = 1.0;
		sampler.anisotropyEnable = VK_FALSE;
	}
	sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	res == vkCreateSampler(context.device, &sampler, nullptr, &state.mesh.textureData.sampler);
	assert(res == VK_SUCCESS);


	// Create image view
	// Textures are not directly accessed by the shaders and
	// are abstracted by image views containing additional
	// information and sub resource ranges
	VkImageViewCreateInfo view = {};
	view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view.viewType = VK_IMAGE_VIEW_TYPE_2D;
	view.format = format;
	view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
	// The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
	// It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
	view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	view.subresourceRange.baseMipLevel = 0;
	view.subresourceRange.baseArrayLayer = 0;
	view.subresourceRange.layerCount = 1;
	// Linear tiling usually won't support mip maps
	// Only set mip map count if optimal tiling is used
	view.subresourceRange.levelCount = 1;
	// The view will be based on the texture's image
	view.image = state.mesh.textureData.image;
	res == vkCreateImageView(context.device, &view, nullptr, &state.mesh.textureData.view);
	assert(res == VK_SUCCESS);

	state.mesh.textureData.descriptor.imageView = state.mesh.textureData.view;
	state.mesh.textureData.descriptor.sampler = state.mesh.textureData.sampler;
	state.mesh.textureData.descriptor.imageLayout = state.mesh.textureData.imageLayout;
}

void setupDescriptorSets(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;

	// Allocate a new descriptor set from the global descriptor pool
	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = context.descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &state.mesh.descriptorSetLayout;

	res = vkAllocateDescriptorSets(context.device, &allocInfo, &state.mesh.descriptorSet);
	assert(res == VK_SUCCESS);

	std::array<VkWriteDescriptorSet, 3> writeDescriptorSet = {};

	// Binding 0 : Uniform buffer VS
	writeDescriptorSet[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet[0].dstSet = state.mesh.descriptorSet;
	writeDescriptorSet[0].descriptorCount = 1;
	writeDescriptorSet[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSet[0].pBufferInfo = &state.mesh.uniformBuffer[0].descriptor;
	// Binds this uniform buffer to binding point 0
	writeDescriptorSet[0].dstBinding = 0;

	// Binding 1 : Uniform buffer FS
	writeDescriptorSet[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet[1].dstSet = state.mesh.descriptorSet;
	writeDescriptorSet[1].descriptorCount = 1;
	writeDescriptorSet[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSet[1].pBufferInfo = &state.mesh.uniformBuffer[1].descriptor;
	// Binds this uniform buffer to binding point 1
	writeDescriptorSet[1].dstBinding = 1;

	// Binding 2 : Object Texture
	writeDescriptorSet[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet[2].dstSet = state.mesh.descriptorSet;
	writeDescriptorSet[2].descriptorCount = 1;
	writeDescriptorSet[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writeDescriptorSet[2].pImageInfo = &state.mesh.textureData.descriptor;
	// Binds this uniform buffer to binding point 1
	writeDescriptorSet[2].dstBinding = 2;

	// Execute the writes to update descriptors for this set
	// Note that it's also possible to gather all writes and only run updates once, even for multiple sets
	// This is possible because each VkWriteDescriptorSet also contains the destination set to be updated
	// For simplicity we will update once per set instead

	vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, nullptr);

}

void setupDescriptorPool(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;

	/*
	Descriptor pool

	Actual descriptors are allocated from a descriptor pool telling the driver what types and how many
	descriptors this application will use

	An application can have multiple pools (e.g. for multiple threads) with any number of descriptor types
	as long as device limits are not surpassed

	It's good practice to allocate pools with actually required descriptor types and counts
	*/

	// We need to tell the API the number of max. requested descriptors per type
	VkDescriptorPoolSize typeCounts[3];

	typeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	typeCounts[0].descriptorCount = 1;

	typeCounts[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	typeCounts[1].descriptorCount = 1;

	typeCounts[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	typeCounts[2].descriptorCount = 1;

	//	// Create the global descriptor pool
	// All descriptors used in this example are allocated from this pool
	VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.pNext = nullptr;
	descriptorPoolInfo.poolSizeCount = 3;
	descriptorPoolInfo.pPoolSizes = typeCounts;
	// Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
	descriptorPoolInfo.maxSets = 3;

	res = (vkCreateDescriptorPool(context.device, &descriptorPoolInfo, nullptr, &context.descriptorPool));
	assert(res == VK_SUCCESS);


}

// Setup layout of descriptors used in this example
// Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
// So every shader binding should map to one descriptor set layout binding
void setupDescriptorSetLayout(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;
	/*

	Descriptor set layout
	The layout describes the shader bindings and types used for a certain descriptor layout and as such must match the shader bindings

	Shader bindings used in this example:

	VS:
		layout (set = 0, binding = 0) uniform uniformVS ...

	FS :
		layout (set = 0, binding = 1) uniform uniformFS ...;
		layout (set = 0, binding = 2) uniform sampler2D ...;
	*/

	// Binding 0: Uniform buffer (Vertex shader)

	std::array<VkDescriptorSetLayoutBinding, 3>layoutBinding = {};
	layoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layoutBinding[0].descriptorCount = 1;
	layoutBinding[0].binding = 0;
	layoutBinding[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	layoutBinding[0].pImmutableSamplers = nullptr;

	// Binding 1: Uniform buffer (Fragment shader)
	layoutBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layoutBinding[1].descriptorCount = 1;
	layoutBinding[1].binding = 1;
	layoutBinding[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	layoutBinding[1].pImmutableSamplers = nullptr;

	// Binding 2: Uniform buffer (Fragment shader - Image Sampler)
	layoutBinding[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	layoutBinding[2].descriptorCount = 1;
	layoutBinding[2].binding = 2;
	layoutBinding[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	layoutBinding[2].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
	descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorLayout.pNext = nullptr;
	descriptorLayout.bindingCount = static_cast<uint32_t>(layoutBinding.size());
	descriptorLayout.pBindings = layoutBinding.data();

	res = vkCreateDescriptorSetLayout(context.device, &descriptorLayout, nullptr, &state.mesh.descriptorSetLayout);
	assert(res == VK_SUCCESS);

}

void updateUniformBuffers(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;
	uint8_t* pData;

	state.mesh.uniformFS.lightPos = glm::vec3(1.0, -0.5, 0.0);
	state.mesh.uniformFS.ambientStrenght = 0.1;
	state.mesh.uniformFS.specularStrenght = 0.5;

	state.mesh.uniformVS.modelMatrix = glm::mat4(1.0f);
	state.mesh.uniformVS.projectionMatrix = glm::perspective(glm::radians(60.0f), (float)context.width / (float)context.height, 0.1f, 256.0f);
	state.mesh.uniformVS.viewMatrix = glm::lookAt(glm::vec3(eyex, eyey, eyez),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f));

	state.mesh.uniformVS.modelMatrix = glm::translate(state.mesh.uniformVS.modelMatrix, glm::vec3(1.0, 0.0, 0.0));
	state.mesh.uniformVS.modelMatrix = glm::rotate(state.mesh.uniformVS.modelMatrix, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	state.mesh.uniformVS.modelMatrix = glm::rotate(state.mesh.uniformVS.modelMatrix, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	state.mesh.uniformVS.modelMatrix = glm::rotate(state.mesh.uniformVS.modelMatrix, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

	// Map uniform buffer and update it
	res = (vkMapMemory(context.device, state.mesh.uniformBuffer[0].memory, 0, sizeof(state.mesh.uniformVS), 0, (void**)&pData));
	memcpy(pData, &state.mesh.uniformVS, sizeof(state.mesh.uniformVS));
	// Unmap after data has been copied
	// Note: Since we requested a host coherent memory type for the uniform buffer, the write is instantly visible to the GPU
	vkUnmapMemory(context.device, state.mesh.uniformBuffer[0].memory);

	res = (vkMapMemory(context.device, state.mesh.uniformBuffer[1].memory, 0, sizeof(state.mesh.uniformFS), 0, (void**)&pData));
	memcpy(pData, &state.mesh.uniformFS, sizeof(state.mesh.uniformFS));
	// Unmap after data has been copied
	// Note: Since we requested a host coherent memory type for the uniform buffer, the write is instantly visible to the GPU
	vkUnmapMemory(context.device, state.mesh.uniformBuffer[1].memory);
}

void prepareUniformBuffers(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;

	VkMemoryRequirements memReq = {};
	VkBufferCreateInfo uniformBufferCreateInfo = {};
	uniformBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	uniformBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

	//Buffer for VS
	uniformBufferCreateInfo.size = sizeof(state.mesh.uniformVS);
	res = bindBufferToMem(context,
		uniformBufferCreateInfo,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		state.mesh.uniformBuffer[0].buffer,
		state.mesh.uniformBuffer[0].memory);
	assert(res == VK_SUCCESS);

	state.mesh.uniformBuffer[0].descriptor.buffer = state.mesh.uniformBuffer[0].buffer;
	state.mesh.uniformBuffer[0].descriptor.offset = 0;
	state.mesh.uniformBuffer[0].descriptor.range = sizeof(state.mesh.uniformVS);

	//Buffer for FS
	uniformBufferCreateInfo.size = sizeof(state.mesh.uniformFS);
	res = bindBufferToMem(context,
		uniformBufferCreateInfo,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		state.mesh.uniformBuffer[1].buffer,
		state.mesh.uniformBuffer[1].memory);
	assert(res == VK_SUCCESS);

	state.mesh.uniformBuffer[1].descriptor.buffer = state.mesh.uniformBuffer[1].buffer;
	state.mesh.uniformBuffer[1].descriptor.offset = 0;
	state.mesh.uniformBuffer[1].descriptor.range = sizeof(state.mesh.uniformFS);

	updateUniformBuffers(context, state);

}

void prepareQuads(struct LHContext& context, struct appState& state) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;

	std::vector<Vertex> vertices =
	{
		{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f } },
		{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f } },
		{ { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f },{ 0.0f, 0.0f, 1.0f } },
		{ {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f },{ 0.0f, 0.0f, 1.0f } }
	};

	// Setup indices
	std::vector<uint32_t> indices = { 0,1,2, 2,3,0 };
	indexCount = static_cast<uint32_t>(indices.size());
	state.mesh.i.count = indexCount;

	res = mapIndiciesToGPU(context, indices.data(), indices.size() * sizeof(uint32_t), state.mesh.i.buffer, state.mesh.i.memory);
	assert(res == VK_SUCCESS);
	res = mapVerticiesToGPU(context, vertices.data(), vertices.size() * sizeof(Vertex), state.mesh.v.buffer, state.mesh.v.memory);
	assert(res == VK_SUCCESS);

	state.mesh.vertexInputBinding = {};
	state.mesh.vertexInputBinding.binding = 0;
	state.mesh.vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	state.mesh.vertexInputBinding.stride = sizeof(Vertex);

	// Inpute attribute bindings describe shader attribute locations and memory layouts
	// These match the following shader layout
	//	layout (location = 0) in vec3 inPos;
	//	layout (location = 1) in vec2 inUV;
	//	layout (location = 2) in vec3 inNormal;
	state.mesh.vertexInputAttributs[0].binding = 0;
	state.mesh.vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.mesh.vertexInputAttributs[0].location = 0;
	state.mesh.vertexInputAttributs[0].offset = offsetof(Vertex, pos);

	state.mesh.vertexInputAttributs[1].binding = 0;
	state.mesh.vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.mesh.vertexInputAttributs[1].location = 1;
	state.mesh.vertexInputAttributs[1].offset = offsetof(Vertex, uv);

	state.mesh.vertexInputAttributs[2].binding = 0;
	state.mesh.vertexInputAttributs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.mesh.vertexInputAttributs[2].location = 2;
	state.mesh.vertexInputAttributs[2].offset = offsetof(Vertex, normal);
}

void loadMesh(struct LHContext& context, struct appState& state, std::string filename) {
	VkResult U_ASSERT_ONLY res;
	bool U_ASSERT_ONLY pass;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str())) {
		throw std::runtime_error(warn + err);
	}

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex = {};

			vertex.pos[0] = attrib.vertices[3*index.vertex_index + 0];
			vertex.pos[1] = attrib.vertices[3*index.vertex_index + 1];
			vertex.pos[2] = attrib.vertices[3*index.vertex_index + 2];

			vertex.uv[0] = attrib.texcoords[2*index.texcoord_index + 0];
			vertex.uv[1] = attrib.texcoords[2*index.texcoord_index + 1];

			vertex.normal[0] = attrib.normals[3*index.normal_index + 0];
			vertex.normal[1] = attrib.normals[3*index.normal_index + 1];
			vertex.normal[2] = attrib.normals[3*index.normal_index + 2];

			vertices.push_back(vertex);
			indices.push_back(indices.size());
		}
	}

	indexCount = static_cast<uint32_t>(indices.size());
	state.mesh.i.count = indexCount;

	res = mapIndiciesToGPU(context, indices.data(), indices.size() * sizeof(uint32_t), state.mesh.i.buffer, state.mesh.i.memory);
	assert(res == VK_SUCCESS);
	res = mapVerticiesToGPU(context, vertices.data(), vertices.size() * sizeof(Vertex), state.mesh.v.buffer, state.mesh.v.memory);
	assert(res == VK_SUCCESS);

	state.mesh.vertexInputBinding = {};
	state.mesh.vertexInputBinding.binding = 0;
	state.mesh.vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	state.mesh.vertexInputBinding.stride = sizeof(Vertex);

	// Inpute attribute bindings describe shader attribute locations and memory layouts
	// These match the following shader layout
	//	layout (location = 0) in vec3 inPos;
	//	layout (location = 1) in vec2 inUV;
	//	layout (location = 2) in vec3 inNormal;
	state.mesh.vertexInputAttributs[0].binding = 0;
	state.mesh.vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.mesh.vertexInputAttributs[0].location = 0;
	state.mesh.vertexInputAttributs[0].offset = offsetof(Vertex, pos);

	state.mesh.vertexInputAttributs[1].binding = 0;
	state.mesh.vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.mesh.vertexInputAttributs[1].location = 1;
	state.mesh.vertexInputAttributs[1].offset = offsetof(Vertex, uv);

	state.mesh.vertexInputAttributs[2].binding = 0;
	state.mesh.vertexInputAttributs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
	state.mesh.vertexInputAttributs[2].location = 2;
	state.mesh.vertexInputAttributs[2].offset = offsetof(Vertex, normal);


}

//#ifdef OBJ_MESH
//void prepareVertices(struct LHContext& context, struct appState& state, int index, bool useStagingBuffers) {
//	VkResult U_ASSERT_ONLY res;
//	bool U_ASSERT_ONLY pass;
//
//	float* vertices;
//	float* normals;
//	uint32_t* indices;
//	std::vector<tinyobj::shape_t> shapes;
//	std::vector<tinyobj::material_t> materials;
//	int nv;
//	int nn;
//	int ni;
//	int i;
//
//	VkBufferCreateInfo buf_info = {};
//	VkMemoryRequirements mem_reqs;
//	VkMemoryAllocateInfo alloc_info = {};
//	uint8_t* pData;
//
//	std::string err = tinyobj::LoadObj(shapes, materials, "data/cube.obj", 0);
//
//	if (!err.empty()) {
//		std::cerr << err << std::endl;
//		return;
//	}
//
//	/*  Retrieve the vertex coordinate data */
//
//	nv = (int)shapes[0].mesh.positions.size();
//	vertices = new GLfloat[nv];
//	for (i = 0; i < nv; i++) {
//		vertices[i] = shapes[0].mesh.positions[i];
//	}
//
//	/*  Retrieve the vertex normals */
//
//	nn = (int)shapes[0].mesh.normals.size();
//	normals = new GLfloat[nn];
//	for (i = 0; i < nn; i++) {
//		normals[i] = shapes[0].mesh.normals[i];
//	}
//
//	/*  Retrieve the triangle indices */
//
//	ni = (int)shapes[0].mesh.indices.size();
//	triangles = ni / 3;
//	indices = new uint32_t[ni];
//	for (i = 0; i < ni; i++) {
//		indices[i] = shapes[0].mesh.indices[i];
//	}
//
//
//	state.cubes[index].vBuffer = new float[nv + nn];
//	int k = 0;
//	for (i = 0; i < nv / 3; i++) {
//		state.cubes[index].vBuffer[k++] = vertices[3 * i];
//		state.cubes[index].vBuffer[k++] = vertices[3 * i + 1];
//		state.cubes[index].vBuffer[k++] = vertices[3 * i + 2];
//		state.cubes[index].vBuffer[k++] = normals[3 * i];
//		state.cubes[index].vBuffer[k++] = normals[3 * i + 1];
//		state.cubes[index].vBuffer[k++] = normals[3 * i + 2];
//	}
//
//	uint32_t dataSize = (nv + nn) * sizeof(state.cubes[index].vBuffer[0]);
//	uint32_t dataStride = 6 * (sizeof(float));
//
//	state.cubes[index].i.count = ni;
//
//	mapIndiciesToGPU(context, indices, sizeof(indices[0]) * ni, state.cubes[index].i.buffer, state.cubes[index].i.memory);
//	mapVerticiesToGPU(context, state.cubes[index].vBuffer, dataSize, state.cubes[index].v.buffer, state.cubes[index].v.memory);
//
//	//// Vertex input descriptions 
//	//// Specifies the vertex input parameters for a pipeline
//
//	//// Vertex input binding
//	//// This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
//
//	state.vertexInputBinding.binding = 0;
//	state.vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
//	state.vertexInputBinding.stride = dataStride;
//
//
//	// Inpute attribute bindings describe shader attribute locations and memory layouts
//	// These match the following shader layout
//	//	layout (location = 0) in vec3 inPos;
//	//	layout (location = 1) in vec3 inNormal;
//	// Attribute location 0: Position
//	//// Attribute location 1: Normal
//	state.vertexInputAttributs[0].binding = 0;
//	state.vertexInputAttributs[0].location = 0;
//	state.vertexInputAttributs[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
//	state.vertexInputAttributs[0].offset = 0;
//	state.vertexInputAttributs[1].binding = 0;
//	state.vertexInputAttributs[1].location = 1;
//	state.vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
//	state.vertexInputAttributs[1].offset = sizeof(float);
//
//}
//#endif // OBJ_MESH


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

int main() {
	theta = 1.5;
	phi = 1.5;
	r = 10.0;

	eyey = 2.0;
	eyez = 7.0;

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
	prepareShader(context, state, "./shaders/shader.vert", "./shaders/shader.frag");
	prepareTexture2(context, state, "data/crate1.jpg");
	//prepareQuads(context, state);
	loadMesh(context, state, "./data/cube.obj");
	prepareUniformBuffers(context, state);
	setupDescriptorSetLayout(context, state);
	setupDescriptorPool(context, state);
	setupDescriptorSets(context, state);
	preparePipelines(context, state);
	buildCommandBuffer(context, state);

	//preparePipelines(context, state);
	//buildCommandBuffers(context, state);

	glfwSetKeyCallback(context.window, key_callback);
	renderLoop(context, state);

	return 0;
}