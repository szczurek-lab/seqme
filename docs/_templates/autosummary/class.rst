{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
   .. automethod:: __init__

   {% if '__call__' in members %}
   .. automethod:: __call__
   {% endif %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      ~{{ name }}.__init__
      {% if '__call__' in members %}
      ~{{ name }}.__call__
      {% endif %}
      {% for item in methods if item != "__init__" %}
      ~{{ name }}.{{ item }}
      {% endfor %}
   {% endif %}
{% endblock %}


{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
