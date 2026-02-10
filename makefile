CC = nvcc
CFLAGS = -I/usr/local/opencv/include/opencv4
LDFLAGS = -L/usr/local/opencv/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui

all: image_processor

image_processor: image_processor.cu
	$(CC) $(CFLAGS) image_processor.cu -o image_processor $(LDFLAGS)

clean:
	rm -f image_processor
