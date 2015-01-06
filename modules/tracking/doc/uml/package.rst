General diagram
===============

.. uml::

  ..@startuml
  package "Tracker"
  package "TrackerFeature"
  package "TrackerSampler"
  package "TrackerModel"

  Tracker -> TrackerModel: create
  Tracker -> TrackerSampler: create
  Tracker -> TrackerFeature: create
  ..@enduml
