#ifndef CVVISUAL_ZOOMALBE_PROXY_OBJECT
#define CVVISUAL_ZOOMALBE_PROXY_OBJECT

#include <QGraphicsProxyWidget>
#include <QGraphicsSceneContextMenuEvent>

#include "../zoomableimage.hpp"

namespace cvv
{
namespace qtutil
{
namespace structures
{
/**
 * @brief spezific class for MatchScene
 */
class ZoomableProxyObject : public QGraphicsProxyWidget
{
      public:
	ZoomableProxyObject(ZoomableImage *zoom);

    ~ZoomableProxyObject() CV_OVERRIDE
	{
	}

      protected:
    virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event) CV_OVERRIDE
	{
		event->ignore();
	}

    virtual void wheelEvent(QGraphicsSceneWheelEvent *event) CV_OVERRIDE;

      private:
	ZoomableImage *image_;
};

}
}
}
#endif
