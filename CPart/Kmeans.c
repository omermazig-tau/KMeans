#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TRUE 1
#define FALSE 0


int isNumber(char str[]);


int isNumber(char str[]) {
    for(int i = 0; i < strlen(str); i++) {
        if(str[i] < '0' || str[i] > '9' ) {
            return FALSE;
        }
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    // Get data from command line
    int k;
    char *strK = NULL;
    char *strIter = NULL;
    int iterations = 200;
    char *input_file;
    char *output_file;
    char str[100];

    if (argc < 4 || argc > 5) {
        printf("Invalid Input!");
        return 1;
    }
    strK = argv[1];
    strIter = argv[2];
    if (argc == 5) {
        if(isNumber(strIter) == FALSE) {
            printf("Invalid Input!");
            return 1;
        }
        iterations = atoi(argv[2]);
        input_file = argv[3];
        output_file = argv[4];
    } else {
        input_file = argv[2];
        output_file = argv[3];
    }
    if(isNumber(strK) == FALSE) {
        printf("Invalid Input!");
        return 1;
    }
    k = atoi(argv[1]);
    //End getting data from command line

    //Read file
    FILE *f;
    int length;
    char * dataPointsStr;

    f = fopen(input_file, "r");
    if(f) {
        fseek(f, 0, SEEK_END);
        length = ftell(f);
        fseek(f, 0, SEEK_SET);
        dataPointsStr = malloc(length);
        if(dataPointsStr) {
            fread(dataPointsStr, 1, length, f);
        }

        //Mine exstra
        char *str = dataPointsStr;
        int countElements = 1;
        int findCountElements = FALSE;
        int countRows = 0;
        for(int i = 0; *str != '\0'; str++) {
            if (*str == ',') {
                countElements++;
            }
        }
    }
    else {
        printf("An Error Has Occurred");
    }





}