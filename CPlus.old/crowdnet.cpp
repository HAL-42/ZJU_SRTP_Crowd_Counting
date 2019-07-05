#include "crowdnet.h"

int main() {

	Net net = dnn::readNetFromCaffe(model_txt, model_bin);
	if (net.empty())
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		exit(-1);
	}
	clog << "net read successfully" << endl;

	const String raw_img_path = "D:\\Win\\Doc\\#Project\\SRTP\\dataset\\UCF_CC_50\\1.jpg";
	Mat raw_img = imread(raw_img_path);
	if (raw_img.empty())
	{
		std::cerr << "Can't read image from the file: " << raw_img_path << std::endl;
		exit(-1);
	}
	clog << "image read sucessfully" << endl;

	raw_img.convertTo(raw_img, CV_32FC3);
	resize(raw_img, raw_img, Size(255, 255));
	Mat blob_img = blobFromImage(raw_img, 1.0, Size(255, 255));
	Mat blob_density;

	CV_TRACE_REGION("forward");
	net.setInput(blob_img, "data");
	blob_density = net.forward("relu1_2");
	Mat tmp1 = net.forward("pool1");


	//int tmp = (int)blob_density.end - (int)blob_density.datastart;
	
	Mat img_density;
	return 0;
}


