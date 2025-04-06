#include <iostream>
#include <chrono>
#include <arm_neon.h>

float means[3] = {0.485, 0.456, 0.406};
float stds[3] = {0.229, 0.224, 0.225};

void normalize(float *x, float *x_out, int C, int H, int W)
{
    // C H W
    for (int c = 0; c < C; c++)
    {
        float channel_mean = means[c];
        float channel_std = stds[c];
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                int index = c * H * W + h * W + w;
                x_out[index] = (x[index] - channel_mean) / channel_std;
            }
        }
    }
}


void normalizeV2(float *x, float *x_out, int C, int H, int W)
{
    for (int c = 0; c < C; c++)
    {
        // 对每个通道，使用均值和标准差
        float channelMeans[4] = {-means[c], -means[c], -means[c], -means[c]}; // 使用正的均值
        float channelStds[4] = {1/stds[c], 1/stds[c], 1/stds[c], 1/stds[c]};       // 标准差不变

        float32x4_t vMean = vld1q_f32(channelMeans);   // 加载均值
        float32x4_t vStd = vld1q_f32(channelStds);     // 加载标准差

        int index = 0;

        // 处理每个像素
        for (int i = 0; i < H * W - 4; i += 4)
        {
            index = i + H * W * c;
            float32x4_t vX = vld1q_f32(x + index);    // 加载 4 个像素
            float32x4_t vXSub = vaddq_f32(vX, vMean);  // 减去均值
            float32x4_t vXOut = vmulq_f32(vXSub, vStd); // 除以标准差
            vst1q_f32(x_out + index, vXOut);            // 存储结果
        }

        // 处理剩余的元素
        if (H * W % 4 != 0)
        {
            for (int i = H * W - H * W % 4; i < H * W; i++)
            {
                index = i + H * W * c;
                x_out[index] = (x[index] - means[c]) / stds[c];
            }
        }
    }
}


int main()
{
    int C = 3, H = 300, W = 300;

    // 输入图像：假设是随机生成的
    float x[3 * 300 * 300] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // 输出标准化图像
    float x_out[3 * 300 * 300];
    float x_out_v2[3 * 300 * 300];

    // 调用标准化函数
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < 100; i++)
    {
        normalize(x, x_out, C, H, W);
    }
    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "V1 Time taken: " << duration.count() << " milliseconds" << std::endl;

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; i++)
    {
        normalizeV2(x, x_out_v2, C, H, W);
    }
    end = std::chrono::steady_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "V2 Time taken: " << duration.count() << " milliseconds" << std::endl;

    // 打印标准化后的输出
    std::cout << "Output after normalize:" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << x_out[i] << " ";
        if ((i + 1) % W == 0)
            std::cout << std::endl;
        if ((i + 1) % (H * W) == 0)
            std::cout << "----" << std::endl;
    }

    std::cout << "\n===================================================" << std::endl;

    // 打印标准化后的输出
    std::cout << "Output after normalizeV2:" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << x_out_v2[i] << " ";
        if ((i + 1) % W == 0)
            std::cout << std::endl;
        if ((i + 1) % (H * W) == 0)
            std::cout << "----" << std::endl;
    }

    return 0;
}