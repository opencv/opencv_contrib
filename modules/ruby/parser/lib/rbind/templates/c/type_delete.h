// deletes <%= cname %> and the underlying object
// <%= full_name %> if the struct is owner of it
void <%= cdelete_method %>(<%= cname %> *ptr)
{
    try
    {
        if(ptr->bowner && ptr->obj_ptr)
            delete fromC(ptr);
        ptr->obj_ptr = NULL;
    }
    catch(std::exception &error){strncpy(&last_error_message[0],error.what(),255);}
    catch(...){strncpy(&last_error_message[0],"Unknown Exception",255);}
    delete ptr;
}
