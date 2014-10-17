# @private
# @note object wrapping <%= full_name %>
class <%= name %>Struct < FFI::Struct
    layout :version,:uchar,
           :size,:size_t,
           :type_id,:pointer,
           :obj_ptr,:pointer,
           :bowner,:bool
    # auto delete
    def self.release(pointer)
        Rbind::<%= cdelete_method %>_struct(pointer) unless pointer.null?
    rescue => e
        puts e
    end
end

<%= add_doc -%>
class <%= name %>
    extend FFI::DataConverter
    native_type FFI::Type::POINTER

    # @private
    #
    # Returns the *Struct type that Rbind uses to store additional information
    # about the memory used by this object
    #
    # @return [FFI::Struct]
    def self.rbind_struct
        <%= name %>Struct
    end

    # returns a null pointer to the object
    def self.null
        new(<%= name %>Struct.new)
    end

<%= add_constructor_doc -%>
    def self.new(*args)
        if args.first.is_a?(FFI::Pointer) || args.first.is_a?(<%= name %>Struct)
            raise ArgumentError, "too many arguments for creating #{self.name} from Pointer" unless args.size == 1
            return super(args.first)
        end
<%= add_constructor %>
        raise ArgumentError, "no constructor for #{self}(#{args.inspect})"
    end

    # @private
    def self.rbind_to_native(obj,context)
        if obj.is_a? <%= name %>
            obj.__obj_ptr__
        else
            raise TypeError, "expected kind of #{name}, was #{obj.class}"
        end
    end

    # @private
    def self.rbind_from_native(ptr,context)
        <%= name %>.new(ptr)
    end

    # @private
    #
    # Performs the conversion a Ruby representation into the FFI representation
    #
    # @param [Object] obj the Ruby representation
    # @param context necessary but undocumented argument from FFI
    # @return [FFI::Pointer,FFI::AutoPointer]
    def self.to_native(obj,context)
        rbind_to_native(obj,context)
    end

    # @private
    # 
    # Performs the conversion from FFI into the Ruby representation that
    # corresponds to this type
    #
    # @param [FFI::Pointer,FFI::AutoPointer] ptr
    # @param [] context
    # @return [Object]
    def self.from_native(ptr,context)
        rbind_from_native(ptr,context)
    end

    # @private
    attr_reader :__obj_ptr__

    # @private
    def initialize(ptr)
        @__obj_ptr__ = if ptr.is_a? <%= name %>Struct
                           ptr
                       else
                           <%= name %>Struct.new(FFI::AutoPointer.new(ptr,<%= name %>Struct.method(:release)))
                       end
    end

    # @private
    # returns true if the underlying pointer is owner of
    # the real object
    def __owner__?
        @__obj_ptr__[:bowner]
    end

    # @private
    # raises if the underlying pointer is a null pointer
    # the real object
    def __validate_pointer__
        raise "NullPointer" if @__obj_ptr__.null? || @__obj_ptr__[:obj_ptr].address == 0
    end

    # converts <%= name %> into a string by crawling through all its attributes
    def to_s
        <%= add_to_s %>
    end

    # @!group Constants
<%= add_consts %>
    # @!endgroup

    # methods
<%= add_methods %>

    # @!group Specializing
<%= add_specializing %>
    # @!endgroup

    # types
<%= add_types %>

end

