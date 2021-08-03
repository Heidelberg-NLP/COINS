
def update_classification_losses(losses, nums, name, bs, loss):
    if not isinstance(loss, float):
        print(type(loss))
        raise

    nums[name] += bs

    losses[name] += loss * bs


def update_generation_losses(losses, nums, micro, macro, bs, length, loss, s):
    # Update Losses
    nums[macro] += bs

    if isinstance(length, int):
        update_indiv_generation_losses(
            losses, nums, micro, macro, bs, length, loss)
    else:
        if s =="knowledge":
            update_tensor_generation_losses_knowledge(
                    losses, nums, micro, macro, bs, length, loss)
        elif s =="sentence":
            update_tensor_generation_losses_sentence(
                    losses, nums, micro, macro, bs, length, loss)



def update_indiv_generation_losses(losses, nums, micro,
                                   macro, bs, length, loss):
    nums[micro] += bs * length

    batch_loss = loss * bs

    losses[micro] += batch_loss
    losses[macro] += batch_loss / length


def update_tensor_generation_losses(losses, nums, micro,
                                    macro, bs, length, loss):
    nums[micro] += length.sum().item()

    losses[micro] += loss.sum().item()
    losses[macro] += (loss / length.float()).sum().item()


def update_tensor_generation_losses_knowledge(losses_k, nums_k, micro,
                                    macro, bs, length, loss):
    nums_k[micro] += length.sum().item()
    losses_k[micro] += loss.sum().item()
    losses_k[macro] += (loss / length.float()).sum().item()


def update_tensor_generation_losses_sentence(losses_s, nums_s, micro,
                                    macro, bs, length, loss):
    nums_s[micro] += length.sum().item()
    losses_s[micro] += loss.sum().item()
    losses_s[macro] += (loss / length.float()).sum().item()

