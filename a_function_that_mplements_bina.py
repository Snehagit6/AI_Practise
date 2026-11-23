
def binary_search(arr, target):
    """
    Description:
    Binary search algorithm to find the target element in the sorted array.

    Parameters:
    arr (list): The sorted list of elements. The list must be sorted in ascending order.
    target: The target element to search. This can be any comparable object, such as an integer, float, or string.

    Returns:
    int: The index of the target element if found, -1 otherwise.

    Examples:
    >>> binary_search([1, 2, 3, 4, 5], 3)
    2
    >>> binary_search([1, 2, 3, 4, 5], 6)
    -1
    >>> binary_search(['a', 'b', 'c', 'd', 'e'], 'c')
    2

    Edge Cases:
    - The function will not work correctly if the input array is not sorted.
    - The function will not work correctly if the target element is None.
    - The function will not work correctly if the target element is not found in the array.

    Note:
    The time complexity of this function is O(log n), where n is the number of elements in the array. The space complexity is O(1), as it only uses a constant amount of space.
    """
    low, high = 0, len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1

Let's execute the binary search algorithm on the list [5,8,9,16,5,50] to find the index of the number 5.

First, we sort the list:
[5, 5, 8, 9, 16, 50]

Now, we can start the binary search:
low = 0, high = 5
mid = (0 + 5) // 2 = 2

The middle element is 8, which is greater than 5. So, we update high = mid - 1 = 2
low = 0, high = 2
mid = (0 + 2) // 2 = 1

The middle element is 5. This is what we're looking for! So, we return the mid index, which is 1.

The output of the binary search algorithm is:
1