require 'logger'

module Rbind
    module Logger
        attr_accessor :log
        def self.extend_object(o)
            super
            o.log = ::Logger.new(STDOUT)
            #o.log.level = ::Logger::INFO
            o.log.level = ::Logger::WARN
            o.log.progname = o.name
            o.log.formatter = proc do |severity, datetime, progname, msg|
                "#{progname}: #{msg}\n"
            end
        end
    end
    extend ::Rbind::Logger
end
