
module Rbind
    class RDataType < RBase
        attr_accessor :typedef
        attr_accessor :invalid_value
        attr_accessor :cdelete_method
        attr_accessor :check_type

        def initialize(name)
            super
            @invalid_value = 0
            @check_type = true
        end

        def ==(other)
            other.generate_signatures[0] == generate_signatures[0]
        end

        # indicates of the type shall be checked before 
        # casting
        def check_type?
            @check_type
        end

        def cname(value=nil)
            if !value
                if basic_type? && !@cname
                    full_name
                else
                    super
                end
            else
                super
                self
            end
        end

        def typedef(value=nil)
            return @typedef unless value
            @typedef = value
            self
        end

        def typedef?
            !!@typedef
        end

        def template?
            false
        end

        # elementar type of c
        def basic_type?
            true
        end

        # holds other operations, types or consts
        def container?
            false
        end

        def to_raw
            self
        end

        def to_single_ptr
            self.to_ptr
        end

        def to_ptr
            RTypeAnnotation.new(self,:ptr => 1)
        end

        def to_ref
            RTypeAnnotation.new(self,:ref => true)
        end

        def to_const
            RTypeAnnotation.new(self,:const => true)
        end

        def to_ownership(val)
            raise "Cannot set memory owner for none pointer types!"
        end

        def remove_const
            self
        end

        def remove_ownership
            self
        end

        def remove_ref
            self
        end

        def remove_ptr
            self
        end

        # returns true if the type is owner of its memory
        def ownership?
            if ref?
                false
            else
                true
            end
        end

        def raw?
            true
        end

        def const?
            false
        end

        def ptr?
            false
        end

        def ref?
            false
        end
    end
end
