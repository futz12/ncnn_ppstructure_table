//
// Created by icys on 25-2-8.
//
#include <NBCaptureCore/Clipboard.h>
#include <NBOCR/model.h>
#include <NBTable/model.h>

int WinMain(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nShowCmd)
{

    NBCapture::SLANet net;
    NBCapture::PPOCRRecognizer rocr;
    NBCapture::PPOCRDetector docr;
    cv::Mat image = cv::imread("./assets/table.png");

    auto result = net.forward(image);
    std::string last_text = "";

    std::string ret = "<table>";

    for (auto &[index, pos] : result)
    {
        if (index.substr(0, 3) == "<td")
        {
            // Draw box
            cv::Point p1(pos[0], pos[1]);
            cv::Point p2(pos[4], pos[5]);
            // cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
            auto sub_img = image(cv::Rect(p1, p2));
            // pad 640
            auto img_pad = cv::Mat(640, 640, CV_8UC3, cv::Scalar(255, 255, 255));
            sub_img.copyTo(img_pad(cv::Rect(0, 0, sub_img.cols, sub_img.rows)));

            auto boxes = docr.forward(img_pad);
            std::string text;
            for (auto &x : boxes)
            {
                cv::Rect sub_area(x.boxPoint[0], x.boxPoint[2]);
                text += rocr.forward(img_pad(sub_area).clone()).text;
            }
            if (index == "<td></td>")
            {
                // std::cout << "<td>" << text << "</td>";
                ret += std::string("<td>") + text + "</td>";
            }
            else
            {
                // std::cout << "<td";
                ret += "<td";

                last_text = text;
            }
        }
        else if (index == ">")
        {
            // std::cout<< ">" << last_text;
            ret += ">" + last_text;
            last_text = "";
        }
        else
        {
            // std::cout << index;
            ret += index;
        }
    }
    ret += "</table>";

    cv::imshow("result", image);
    cv::waitKey(0);
    std::cout << ret;

    NBCapture::LoadHtml2Clipboard(ret);
    return 0;
}
