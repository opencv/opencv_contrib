package lowgui;

import org.opencv.core.*;

import java.awt.*;
import java.awt.image.*;

import javax.swing.JFrame;


/**
* mock highgui:
*  some diff to the original: 
*   - imshow does not work per name, but per NamedWindow instance
*   - waitKey is totally optional for the blitting, you do not *have* to call it
*   - waitKey(0) does *not* wait at all (waitKey(-1) does).
**/

public class NamedWindow extends JFrame
{
    ShowImage _imshow;
    WaitKey  _waitkey;

    public NamedWindow(String name) {
        super(name);
        setSize(640, 480);
        setVisible(true);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        _imshow = new ShowImage();
        getContentPane().add(_imshow);

        _waitkey = new WaitKey();
        addKeyListener(_waitkey);
    }

    public void imshow(Mat m) {
        _imshow.set(m);
    }

    public int waitKey(int t) {
        return _waitkey.get(t);
    }
}
