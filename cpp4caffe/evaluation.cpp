#define REG_USE_CNN 1
#pragma warning(disable:4996)
#include "mrutil.h"
#include "cnnpredictor.h"
const string errordir = caffeplatedir + "error";
const string platedatadir = caffeplatedir+"data";

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
	vector<string>subdirs=getAllSubdirs(dir);
	for (int i = 0; i < subdirs.size(); i++)
	{
		string subdir = dir + "/" + subdirs[i];
		cleardir(subdir);
	}
}

int evaluation()
{
	string line;
	string label;
	int rightcount = 0, errorcount = 0, total = 0;	
	if (!exist(errordir.c_str()))
	{
		cout << "Error dir not exist" << endl;
		_mkdir(errordir.c_str());
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
			auto ret = split(CnnPredictor::getInstance()->predict(img), " ")[1];
			if (ret == sub)
				rightcount++;
			else
			{
				errorcount++;
				string errorlabeldir = errordir;
				errorlabeldir = errorlabeldir + "/" + sub;
				if (!exist(errorlabeldir.c_str()))
				{
					_mkdir(errorlabeldir.c_str());
				}
				string errorfilepath = errorlabeldir + "/" + file.substr(0,file.size()-4) + "_" + sub + "_" + ret + ".png";
				cout << sub + "/" + file.substr(0, file.size() - 4) + ":" + ret << endl;
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

int main(int argc,char*argv[])
{
	if (argc==1)
		evaluation();
	else
	{
		cv::Mat img = cv::imread(argv[1]);
		cout << CnnPredictor::getInstance()->predict(img) << endl;
	}
	return 0;
}