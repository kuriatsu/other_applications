#include<stdio.h>

int main(int argc, char *argv)
{
	char rgb_order[3] = "RGB";
	int rgb_value[3];
	int hsv_value[3];
	int max_rgb=0, min_rgb=300;
	int max_flag, min_flag;
	
	for (int i = 0; i < 3; i++)
	{
		printf("input %c\n", rgb_order[i]);
		scanf("%d", &rgb_value[i]);

		if (rgb_value[i] <= min_rgb)
		{
			min_rgb = rgb_value[i];
			min_flag = i;
		}

		if (rgb_value[i] >= max_rgb)
		{
			max_rgb = rgb_value[i];
			max_flag = i;
		}
	}
	printf("%d %d %d %d\n", min_flag, max_flag, min_rgb, max_rgb);

	if (max_rgb == 0)
	{
		printf("H : 0, S : 0, V : 0\n");
		return 0;
	}

	hsv_value[2] = max_rgb;

	hsv_value[1] = 255 * (max_rgb - min_rgb) / max_rgb;
	
	if (rgb_value[0] == rgb_value[1] && rgb_value[1] == rgb_value[2])
	{
		printf("H : 0, S : %d, V : %d\n",  hsv_value[1],  hsv_value[2]);
		return 0;
	}
	
	switch(max_flag){
		case 0:
			hsv_value[0] = 60 * (rgb_value[1] - rgb_value[2]) / (max_rgb - min_rgb);
			break;
		case 1:
			hsv_value[0] = 60 * (2 + (rgb_value[2] - rgb_value[0]) / (max_rgb - min_rgb));
			break;
		case 2:
			hsv_value[0] = 60 * (4 + (rgb_value[0] - rgb_value[1]) / (max_rgb - min_rgb));
			break;
	}

	if (hsv_value[0] < 0)
		hsv_value[0] += 360;

	printf("H : %d, S : %d, V : %d\n", hsv_value[0],  hsv_value[1],  hsv_value[2]);

	return 0;
}
