package lowgui;

import org.opencv.core.*;
import java.awt.*;
import java.awt.image.*;

/**
*   a mimimal awt Panel to show cv::Mat in java
**/

public class ShowImage extends Panel {
    BufferedImage  image;

    public ShowImage(Mat m) { set(m); }
    public ShowImage() { set(new Mat(10,10,16,new Scalar(200,0,0))); }

    public void set(Mat m) {
        image = bufferedImage(m);
        repaint();
    }

    public void paint(Graphics g) {
        g.drawImage( image, 0, 0, null);
    }

    public static BufferedImage bufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        m.get(0,0,((DataBufferByte)image.getRaster().getDataBuffer()).getData()); // get all the pixels
        return image;
    }
}
