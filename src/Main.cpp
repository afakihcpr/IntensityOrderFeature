#include <MyDescriptors.h>
#include "Utils.h"

int main(int argc, char* argv[])
{
	char* img_file = NULL;
	char* region_file = NULL;
	char* des_file = NULL;

    iop::Params params;

	/*************************  Parsing the arguments  *****************************/
	int counter=0;
	while( ++counter < argc )
	{
		if( !strcmp("-type", argv[counter] ))
		{
			counter++;

			if(strcmp(argv[counter], "liop") == 0)
			{
                params.des_type = iop::LIOP;
			}
			else if(strcmp(argv[counter], "oiop") == 0)
			{
                params.des_type = iop::OIOP;
			}	
			else if (strcmp(argv[counter], "miop") == 0)
			{
                params.des_type = iop::MIOP;
			}
			else if (strcmp(argv[counter], "miop_fast") == 0)
			{
                params.des_type = iop::MIOP_FAST;
			}
			else
            {	std::cerr << "Invalid type argument!" << std::endl;
            std::cout << "it must be csgp_kd or csgp_go!\n" << std::endl;
			return -1;

			}
			continue;
		}

		if( !strcmp("-img", argv[counter] ))
		{
			img_file = argv[++counter];
			continue;
		}

		if( !strcmp("-region", argv[counter] ))
		{
			region_file = argv[++counter];
			continue;
		}

		if( !strcmp("-des", argv[counter] ))
		{
			des_file = argv[++counter];
			continue;
		}

		if( !strcmp("-initSigma", argv[counter] ))
		{
			params.initSigma= (double)atof(argv[++counter]);
			continue;
		}

		if( !strcmp("-srNum", argv[counter] ))
		{
			params.srNum = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-liopType", argv[counter] ))
		{
			params.liopType = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-liopRegionNum", argv[counter] ))
		{
			params.liopRegionNum = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-liopNum", argv[counter] ))
		{
			params.liopNum = (int)atoi(argv[++counter]);
            iop::GeneratePatternMap(&params.pLiopPatternMap, &params.pLiopPosWeight, params.liopNum);
			if (params.liopNum != 3 && params.liopNum != 4)
			{
                std::cerr << "Invalid command line argument: \"" << argv[counter] <<"\""<< std::endl;
                std::cerr << "The liopNum should be 3 or 4. " << std::endl;
				return -1;
			}
			continue;
		}

		if( !strcmp("-oiopType", argv[counter] ))
		{
			params.oiopType = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-oiopRegionNum", argv[counter] ))
		{
			params.oiopRegionNum = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-oiopQuantLevel", argv[counter] ))
		{
			params.oiopQuantLevel = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-oiopNum", argv[counter] ))
		{
			params.oiopNum = (int)atoi(argv[++counter]);

			if (params.oiopNum != 2 && params.oiopNum != 3)
			{
                std::cerr << "Invalid command line argument: \"" << argv[counter] <<"\""<< std::endl;
                std::cerr << "The oiopNum should be 2 or 3. " << std::endl;
				return -1;

			}
			continue;
		}

		if( !strcmp("-lsRadius", argv[counter] ))
		{
			params.lsRadius = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-normPatchWidth", argv[counter] ))
		{
			params.normPatchWidth = (int)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-nSigma", argv[counter] ))
		{
			params.nSigma = (double)atof(argv[++counter]);
			continue;
		}

		if( !strcmp("-liopThre", argv[counter] ))
		{
			params.liopThre = (double)atof(argv[++counter]);
			continue;
		}

		if( !strcmp("-pcaFile", argv[counter] ))
		{
			params.PCAFile.assign(argv[++counter]);
			continue;
		}

		if( !strcmp("-pcaBasisNum", argv[counter] ))
		{
			params.PCABasisNum = atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-isApplyPCA", argv[counter] ))
		{
			params.isApplyPCA = (int)atoi(argv[++counter]);
			continue;
		}

        std::cerr << "Invalid command line argument: \"" << argv[counter] <<"\""<< std::endl;
		return -1;
	}

	//ensure normPatchWidth is odd
	if(params.normPatchWidth%2 == 0)
		params.normPatchWidth++;

    if ((params.des_type == iop::MIOP || params.des_type == iop::MIOP_FAST) &&  params.isApplyPCA)
	{
        params.desFormat = iop::DES_FLOAT;
	}

    std::vector<iop::AffineKeyPoint> kpts;
    cv::Mat dess;

	//read image and affine regions
    cv::Mat img = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
	if (params.initSigma > 0)
	{
        iop::GaussianBlur(img, img, cv::Size(0, 0), params.initSigma);
	}
    iop::ReadKpts(region_file, kpts);

	//extract descriptors
    iop::MyDescriptors commonDes(params);
	commonDes.compute(img, kpts, dess);
    iop::WriteDess(des_file, kpts, dess, params.desFormat);

	return 0;
}
