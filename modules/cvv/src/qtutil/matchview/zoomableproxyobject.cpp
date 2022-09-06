
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
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
			      event->buttons(), event->modifiers(),
			      Qt::NoScrollPhase, true };
#else
			      event->delta(),   event->orientation(),
			      event->buttons(), event->modifiers() };
#endif
	image_->wheelEvent(&newEvent);
}

}
}
}
