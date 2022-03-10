// coinGenerator.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "modelGen.h"

extern "C" {
int main(int argc, char** argv)
// int main()
{
    argc = 7;
    // char* argv[7];
    /*
    char arg1[] = "src000.jpg";
    char arg2[] = "src000_mask.png";
    char arg3[] = "src000.bin";
    char arg4[] = "src000_text.png";
    char arg5[] = "src000.stl";
    char arg6[] = "src001.png";
    */

    /*
    char arg1[] = "model0056.png";
    char arg2[] = "model0056_mask.png";
    char arg3[] = "model0056.bin";
    char arg4[] = "text11.png";
    char arg5[] = "model0056.stl";
    char arg6[] = "model0057.png";
    */
    /*
    char arg1[] = "002F.png";
    char arg2[] = "002F_mask.png";
    char arg3[] = "002F.bin";
    char arg4[] = "002F_TEXT.png";
    char arg5[] = "002.stl";
    char arg6[] = "002R.png";
    */
    
    // for(int i=0;i<argc;i++){
    //   printf("%s\n", argv[i]);
    // }

    // char arg1[] = "007F.png";
    // char arg2[] = "NONE";
    // char arg3[] = "NONE";
    // char arg4[] = "007F_TEXT.png";
    // //char arg4[] = "empty_TEXT.png";
    // char arg5[] = "007_2.stl";
    // char arg6[] = "003R.png";

    // argv[1] = arg1;
    // argv[2] = arg2;
    // argv[3] = arg3;
    // argv[4] = arg4;
    // argv[5] = arg5;
    // argv[6] = arg6;

    if (argc != 7)
    {
        fprintf(stderr, "ERROR: some argument is missing\n");
        return ERROR_INVALID_ARGUMENTS_NUMBERS;
    }

    modelGen generator;
    std::string strSourceImageFileName;
    std::string strRearImageFileName;
    std::string strMaskImageFileName;
    std::string strNoseDorsumFileName;
    std::string strTextImageFileName;
    std::string str3dModelFileName;

    strSourceImageFileName = argv[1];
    strMaskImageFileName = argv[2];
    strNoseDorsumFileName = argv[3];
    strTextImageFileName = argv[4];

    if (strMaskImageFileName == "NONE" || strNoseDorsumFileName == "NONE")
        generator.setEdgeOnlyMode();

    //########## getting a source image ##########
    generator.setImage(strSourceImageFileName);

    //########## setting a maske image ##########
    generator.setMask(strMaskImageFileName);

    //########## setting a nose dorsum ##########
    generator.setNoseDorsum(strNoseDorsumFileName);

    //########## computing data for height map ##########
    generator.preComputeHeightMap();

    //########## setting a text image ##########
    generator.setTextImage(strTextImageFileName);



    //########## getting a rear image ##########
    strRearImageFileName = argv[6];
    generator.setImageRear(strRearImageFileName);

    
    
    //########## generating a result 3D model ##########
    str3dModelFileName = argv[5];
    generator.generate3Dmodel(str3dModelFileName, 25);
    
    return NO_ERRORS;
}
}