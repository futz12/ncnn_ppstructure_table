//
// Created by icys on 25-2-5.
//

#ifndef MODEL_Table_H
#define MODEL_Table_H

namespace NBCapture {
class SLANet {
public:
    ncnn::Net cnn;
    ncnn::Net slahead;
    std::vector<std::string> token_list;

    SLANet();

    std::vector<std::pair<std::string,std::array<float,8>>> forward(const cv::Mat& image);

};
}

#endif //MODEL_Table_H
