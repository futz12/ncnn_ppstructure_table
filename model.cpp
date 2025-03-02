//
// Created by icys on 25-2-5.
//

#include "model.h"

namespace NBCapture {

SLANet::SLANet() {
    cnn.load_param("./assets/ch_PP-StructrureV2_SLANet_plus_cnn.param");
    cnn.load_model("./assets/ch_PP-StructrureV2_SLANet_plus_cnn.bin");

    slahead.load_param("./assets/ch_PP-StructrureV2_SLANet_plus_slahead.param");
    slahead.load_model("./assets/ch_PP-StructrureV2_SLANet_plus_slahead.bin");

    // ./assets/table_structure_dict_ch.txt
    std::ifstream ifs("./assets/table_structure_dict_ch.txt");
    std::string line;
    token_list.push_back("");
    while (std::getline(ifs, line)) {
        token_list.push_back(line);
    }
}

std::vector<std::pair<std::string,std::array<float,8>>> SLANet::forward(const cv::Mat& image) {
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, 488, 488);
    float mean[] = { 0.485f * 255.0f , 0.456f * 255.0f, 0.406f * 255.0f };
    float std[] =  { 1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f };
    in.substract_mean_normalize(mean, std);

    auto ex = cnn.create_extractor();
    ex.input("in0", in);
    ncnn::Mat feat;
    ex.extract("out0", feat);

    feat = feat.reshape( 96, 256);
    ncnn::Mat hidden(256,1);
    ncnn::Mat one_hot_feat(50);

    // hidden = 0
    hidden.fill(0.0f);
    // one_hot_feat = [1, 0, 0, 0, 0, ...]
    one_hot_feat.fill(0.0f);
    one_hot_feat[0] = 1.0f;

    int step = 0;
    static const int max_step = 256;
    static const int eos = 49;

    std::vector<std::pair<std::string,std::array<float,8>>> result;

    while (step < max_step) {
        auto ex2 = slahead.create_extractor();
        ex2.input("in0", hidden.clone());
        ex2.input("in1", feat.clone());
        ex2.input("in2", one_hot_feat.clone());

        ncnn::Mat hidden2, structure, loc;

        ex2.extract("out0", hidden2);
        ex2.extract("out1", structure);
        ex2.extract("out2", loc);

        hidden = hidden2.clone();

        int token = 0;
        float max_score = -1e30;
        for (int i = 0; i < 50; i++) {
            if (structure[i] > max_score) {
                max_score = structure[i];
                token = i;
            }
        }

        if (token == eos) {
            printf("eos\n");
            break;
        }
        std::string code = token_list[token];
        std::array<float,8> locs;
        // x
        for (int i = 0; i < 8; i+=2) {
            locs[i] = loc[i] * image.cols;
        }
        // y
        for (int i = 1; i < 8; i+=2) {
            locs[i] = loc[i] * image.rows;
        }
        result.push_back(std::make_pair(code, locs));

        one_hot_feat.fill(0.0f);
        one_hot_feat[token] = 1.0f;
        step++;
    }

    return result;
}
}