#include "mrdir.h"
#include "mrutil.h"
#include "mropencv.h"
#include "LenetClassifier.h"
using namespace std;
const string errordir = caffeplatedir + "/error";
const string platedatadir = caffeplatedir + "/data";

void cleardir(const string dir)
{
	vector<string>files=getAllFilesinDir(dir);
	for (int i = 0; i < files.size(); i++)
	{
		string filepath = dir + "/" + files[i];
		remove(filepath.c_str());
	}
}

void clearerror(const string dir)
{
    cout << "clearing" << dir << endl;
	vector<string>subdirs=getAllSubdirs(dir);
	for (int i = 0; i < subdirs.size(); i++)
	{
		string subdir = dir + "/" + subdirs[i];
        cout << subdirs[i]<<endl;
		cleardir(subdir);
	}
    cout << "clearing done"<< endl;
}

int evaluation()
{
	string line;
	string label;
	int rightcount = 0, errorcount = 0, total = 0;	
	if (!EXISTS(errordir.c_str()))
	{
		cout << "Error dir not exist" << endl;
		MKDIR(errordir.c_str());
	}
	clearerror(errordir);
	vector<string>subdirs=getAllSubdirs(platedatadir);
	for (auto sub : subdirs)
	{
		string subdir = platedatadir + "/" + sub;
		vector<string>files=getAllFilesinDir(subdir);
		for (auto file : files)
		{
			string fileapth = subdir + "/" + file;
			cv::Mat img = cv::imread(fileapth);
			auto ret=CLenetClassifier::getInstance()->predict(img).first;
			if (ret == string2int(sub))
				rightcount++;
			else
			{
                cout << sub + "/" + file.substr(0, file.size() - 4) + ":" + int2string(ret) << endl;
				errorcount++;
				string errorlabeldir = errordir;
				errorlabeldir = errorlabeldir + "/" + sub;
				if (!EXISTS(errorlabeldir.c_str()))
				{
					MKDIR(errorlabeldir.c_str());
				}
				string errorfilepath = errorlabeldir + "/" + file.substr(0,file.size()-4) + "_" + sub + "_" + int2string(ret) + ".png";
				imshow("error", img);
				imwrite(errorfilepath, img);
				cv::waitKey(1);
			}
			total++;
		}
	}
	cout << "acc:" << rightcount << "/" << total << endl;
	cout << rightcount*1.0 / total << endl;
	return 0;
}

int testimg(const std::string imgpath = "img/0.jpg")
{
    cv::Mat img = imread(imgpath);
    TickMeter tm;
    tm.start();
    auto p = CLenetClassifier::getInstance()->predict(img);
    tm.stop();
    std::cout << p.first << std::endl;// " " << p.second << endl;
    std::cout << tm.getTimeMilli() << "ms" << std::endl;
    return 0;
}

int testdir(const std::string dir = "img")
{
    auto files = getAllFilesinDir(dir);
    for (int i = 0; i < files.size(); i++)
    {
        std::string imgpath = dir + "/" + files[i];
        std::cout << files[i] << ":";
        testimg(imgpath);
    }
    return 0;
}

int main(int argc,char*argv[])
{
	if (argc==1)
		evaluation();
	else
	{
        testimg();
        testdir();
	}
	return 0;
}