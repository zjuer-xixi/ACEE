#include<stdio.h>
int main(){
	FILE *fp1;
	FILE *fp2;

    int a,b;
    char context1,context2;
    fp1 = fopen ("file1.txt","r");
    fp2 = fopen ("file2.txt","r");
	 while ((a = fgetc(fp1)) != EOF) // 标准C I/O读取文件循环
   { 
       context1+=a;
   }
	 while ((b = fgetc(fp2)) != EOF) // 标准C I/O读取文件循环
   { 
       context2+=b;
   }

	fclose(fp1);
	fclose(fp1);

    if(context1==context2){
    	printf("the code is right"); 
	}
	else{
		printf("the code is wrong");
	}
}

