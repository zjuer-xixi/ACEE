#include <stdio.h>

#define ElementType int
#define MAXN 100

void merge_pass(ElementType list[], ElementType sorted[], int N, int length);

void output(ElementType list[], int N) {
	int i;
	for (i = 0; i < N; i++)
		printf("%d ", list[i]);
	printf("\n");
}

void merge_sort(ElementType list[], int N) {
	ElementType extra[MAXN]; /* the extra space required */
	int length = 1;           /* current length of sublist being merged */
	while (length < N) {
		merge_pass(list, extra, N, length); /* merge list into extra */
		output(extra, N);
		length *= 2;
		merge_pass(extra, list, N, length); /* merge extra back to list */
		output(list, N);
		length *= 2;
	}
}

void merge(ElementType list[], ElementType sorted[], int Lpos, int Rpos, int RightEnd ) {
	int  i, LeftEnd, NumElements, TmpPos;
	LeftEnd = Rpos - 1;
	TmpPos = Lpos;
	NumElements = RightEnd - Lpos + 1;
	while( Lpos <= LeftEnd && Rpos <= RightEnd )
		if ( list[ Lpos ] <= list[ Rpos ] )
			sorted[ TmpPos++ ] = list[ Lpos++ ];
		else
			sorted[ TmpPos++ ] = list[ Rpos++ ];
	while( Lpos <= LeftEnd )
		sorted[ TmpPos++ ] = list[ Lpos++ ];
	while( Rpos <= RightEnd )
		sorted[ TmpPos++ ] = list[ Rpos++ ];
	for( i = 0; i < NumElements; i++, RightEnd--)
		list[ RightEnd ] = sorted[ RightEnd ];
}

void merge_pass( ElementType list[], ElementType sorted[], int N, int length ) {
	int i, ls, rs, re;
	for (i = 0; i <= N - 2 * length; i += 2 * length) {
		ls = i;
		rs = i + length;
		re = i + 2 * length - 1;
		merge(list, sorted,ls, rs, re);
	}

	if(i + length >= N) {
		for (int j = i; j < N; j++)
			sorted[j] = list[j];
	}

	else {
		ls = i;
		rs = i + length;
		re = N - 1;
		merge(list, sorted, ls, rs, re);
	}


}

int main() {
	int N, i;
	ElementType A[MAXN];

	scanf("%d", &N);
	for (i = 0; i < N; i++)
		scanf("%d", &A[i]);
	merge_sort(A, N);
	output(A, N);

	return 0;
}

