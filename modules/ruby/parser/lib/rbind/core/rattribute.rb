module Rbind
    class RAttribute < RBase
        attr_accessor :type

        def initialize(name,type)
            super(name)
            raise ArgumentError,"no type" unless type
            @type = type
            @readable = true
            @writable = false
        end

        def ==(other)
            if other
                type == other.type
            else
                false
            end
        end

        def generate_signatures
            @type.generate_signatures.map do |s|
                "#{s} #{name}"
            end
        end

        def readable!(value = true)
            @readable = value
            self
        end

        def writeable!(value = true)
            @writeable = value
            self
        end

        def readable?
            !!@readable
        end

        def writeable?
            !!@writeable
        end
    end
end
