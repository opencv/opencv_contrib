
#include <QWheelEvent>

#include "zoomableproxyobject.hpp"

namespace cvv
{
namespace qtutil
{
namespace structures
{

ZoomableProxyObject::ZoomableProxyObject(ZoomableImage *zoom)
    : QGraphicsProxyWidget{}, image_{ zoom }
{
	QGraphicsProxyWidget::setWidget(image_);
}

void ZoomableProxyObject::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	QPoint delta{ event->delta(), 0 };
	QWheelEvent newEvent{ event->pos(),     event->screenPos(),
			      delta,            delta,
			      event->buttons(), event->modifiers(),
			      Qt::NoScrollPhase, true };
	image_->wheelEvent(&newEvent);
}

}
}
}
